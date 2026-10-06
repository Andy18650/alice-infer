// The alice model format loader (docs/model-format.md RC1,
// docs/model-loader.md §8 scope). init_engine is the public entry: it
// reads the .model envelope (magic/framing/signature-state), extracts the
// signed metadata + non-device members (tokenizer, chat_template), and loads
// one gpu::Model per tensor-parallel rank. The Engine then owns the worker
// threads and the shared work barrier.
// Signature verification is deferred with the chain of trust: the
// envelope's signature state is DECLARED loudly, not checked.

#include "engine.hpp"

#include "gpu/model.hpp"  // per-rank Model + WorkItem

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <rccl/rccl.h>

#include "data_loader.hpp"  // LocalFileDataLoader (content-addressed fetch)
#include "sampler.hpp"      // host-side Sampler

namespace alice {

using nlohmann::json;

namespace {  // private helpers

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("model loader: " + what);
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

// Reads a .model descriptor file and returns the parsed envelope: text
// region (everything before the first NUL, which must be \n- or \r\n-
// terminated; a binary tail may follow in single-file models, so no
// whole-file read), {"ALICE" magic, then the json parse.
json read_envelope(const std::filesystem::path& file) {
  constexpr size_t kMaxDescriptor = 1 << 20;

  std::ifstream in(file, std::ios::binary);
  if (!in) fail("cannot open '" + file.string() + "'");
  std::string text(kMaxDescriptor, '\0');
  in.read(text.data(), kMaxDescriptor);
  const size_t got = size_t(in.gcount());

  if (got < 8 || text.compare(0, 8, "{\"ALICE\"") != 0)
    fail("bad magic: a .model file must begin with {\"ALICE\"");

  const size_t nul = text.find('\0');
  if (nul == std::string::npos || nul >= got)
    fail("'" + file.string() + "' has no NUL within the first 1 MiB — "
         "not a .model file?");
  if (nul == 0 || text[nul - 1] != '\n')
    fail("text region must end with \\n before the NUL gate");
  size_t end = nul - 1;
  if (end > 0 && text[end - 1] == '\r') --end;
  text.resize(end);

  json envelope;
  try {
    envelope = json::parse(text);
  } catch (const json::exception& e) {
    fail(std::string("envelope JSON: ") + e.what());
  }
  return envelope;
}

}  // namespace

namespace engine {

// Private threading state (non-movable members live here).
struct Engine::ThreadState {
  std::mutex work_mutex;
  std::condition_variable work_cv;
  gpu::WorkItem work;
  uint64_t work_generation = 0;
  std::vector<uint64_t> worker_generations;
  std::atomic<int> pending{0};

  std::mutex done_mutex;
  std::condition_variable done_cv;

  bool shutdown = false;

  std::mutex load_mutex;
  std::condition_variable load_cv;
  int loaded_count = 0;
  std::exception_ptr load_error;
};

Engine::Engine() : ts_(std::make_unique<ThreadState>()) {}

Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Engine::~Engine() {
  if (!ts_) return;

  // Signal shutdown to every joinable worker and join.
  {
    std::lock_guard<std::mutex> lock(ts_->work_mutex);
    ts_->shutdown = true;
  }
  ts_->work_cv.notify_all();

  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
}

void Engine::submit_work(const gpu::WorkItem& work) {
  auto& ts = *ts_;
  {
    std::lock_guard<std::mutex> lock(ts.work_mutex);
    ts.work = work;
    ts.work_generation++;
    ts.pending = static_cast<int>(models_.size());
  }
  ts.work_cv.notify_all();

  std::unique_lock<std::mutex> lock(ts.done_mutex);
  ts.done_cv.wait(lock, [&] { return ts.pending == 0; });

  for (auto& m : models_) {
    if (auto e = m->error()) std::rethrow_exception(e);
  }
}

void worker_main(Engine* engine, int rank, int world_size, gpu::Model* model,
                 const Descriptor& descriptor,
                 const loader::LocalFileDataLoader& data_loader,
                 LoadStats& st) {
  auto& ts = *engine->ts_;

  // Load this rank's weights.
  try {
    model->load(rank, world_size, descriptor, data_loader, st);
  } catch (...) {
    model->set_error(std::current_exception());
    {
      std::lock_guard<std::mutex> lock(ts.load_mutex);
      ts.loaded_count++;
    }
    ts.load_cv.notify_one();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(ts.load_mutex);
    ts.loaded_count++;
  }
  ts.load_cv.notify_one();

  // Work loop.
  while (true) {
    gpu::WorkItem work;
    {
      std::unique_lock<std::mutex> lock(ts.work_mutex);
      ts.work_cv.wait(lock, [&] {
        return ts.shutdown ||
               ts.worker_generations[rank] != ts.work_generation;
      });
      if (ts.shutdown) break;
      work = ts.work;
      ts.worker_generations[rank] = ts.work_generation;
    }

    model->execute(work);

    if (--ts.pending == 0) {
      std::lock_guard<std::mutex> lock(ts.done_mutex);
      ts.done_cv.notify_one();
    }
  }
}

Engine init_engine_impl(const std::filesystem::path& model_file,
                        const ModelConfig& mc, const SamplingConfig& sc, LoadStats& st) {
  Engine out;
  auto& ts = *out.ts_;

  Timer t;
  const json envelope = read_envelope(model_file);
  {
    if (!envelope.is_object()) fail("envelope must be an object");
    const json& alice = envelope.at("ALICE");
    if (!alice.is_object()) fail("envelope.ALICE must be an object");
    if (alice.at("format_version").get<uint64_t>() != 1)
      fail("envelope.ALICE: unsupported format_version (this loader reads 1)");
    if (envelope.contains("signature") && !envelope.at("signature").is_null()) {
      std::string key = "?", alg = "?";
      const json& s = envelope.at("signature");
      if (s.is_object()) {
        if (s.contains("key")) key = s.at("key").get<std::string>();
        if (s.contains("alg")) alg = s.at("alg").get<std::string>();
      }
      std::fprintf(stderr,
                   "[model loader] descriptor carries a signature (alg=%s key=%s) — "
                   "verification not implemented (chain-of-trust round); proceeding\n",
                   alg.c_str(), key.c_str());
    } else {
      std::fprintf(stderr,
                   "[model loader] unsigned descriptor — proceeding (dev artifacts only)\n");
    }
  }

  const json& model = envelope.at("model");
  if (!model.is_object()) fail("envelope.model must be an object");
  Descriptor descriptor = model.get<Descriptor>();
  if (descriptor.version != 1)
    fail("descriptor: unsupported version " + std::to_string(descriptor.version) +
         " (this loader reads version 1)");
  st.parse_ms = t.ms();
  st.specs = descriptor.specs.size();

  const std::filesystem::path base = model_file.parent_path();
  const loader::LocalFileDataLoader data_loader(envelope.at("files"),
                                                base.string());

  auto fetch_string = [&](const char* key) -> std::string {
    const auto it = descriptor.members.find(key);
    if (it == descriptor.members.end()) return {};
    std::string s;
    data_loader.fetch(it->second.id, [&](std::span<const std::byte> blob) {
      s.assign(reinterpret_cast<const char*>(blob.data()), blob.size());
    });
    return s;
  };
  out.chat_template_ = fetch_string("chat_template");

  uint32_t bos_token_id = 0;
  std::vector<uint32_t> eos_token_ids;
  std::string tokenizer_json;
  if (const auto it = descriptor.members.find("tokenizer");
      it != descriptor.members.end()) {
    data_loader.fetch(it->second.id, [&](std::span<const std::byte> blob) {
      tokenizer_json.assign(reinterpret_cast<const char*>(blob.data()), blob.size());
    });
    const json& tc = it->second.config;
    if (tc.contains("bos_token_id"))
      bos_token_id = tc.at("bos_token_id").get<uint32_t>();
    if (tc.contains("eos_token_id")) {
      const json& e = tc.at("eos_token_id");
      if (e.is_array())
        for (const auto& x : e) eos_token_ids.push_back(x.get<uint32_t>());
      else
        eos_token_ids.push_back(e.get<uint32_t>());
    }
  }
  out.tokenizer_ = frontend::Tokenizer::from_json(tokenizer_json, bos_token_id,
                                                  std::move(eos_token_ids));

  out.name_ = descriptor.name;
  out.producer_ = descriptor.producer;
  out.packager_ = descriptor.packager;
  out.license_ = descriptor.license;
  out.license_url_ = descriptor.license_url;
  out.license_text_ = descriptor.license_text;
  out.max_context_ = descriptor.max_context;

  // Device models: one per rank, each loaded in its own worker thread.
  Timer load_timer;

  out.models_.reserve(mc.tp_degree);
  out.workers_.reserve(mc.tp_degree);
  ts.worker_generations.assign(mc.tp_degree, 0);

  for (uint32_t r = 0; r < mc.tp_degree; ++r) {
    out.models_.push_back(std::make_unique<gpu::Model>(mc));
    gpu::Model* m = out.models_.back().get();
    out.workers_.emplace_back(worker_main, &out, r, static_cast<int>(mc.tp_degree),
                              m, std::cref(descriptor), std::cref(data_loader),
                              std::ref(st));
  }

  {
    std::unique_lock<std::mutex> lock(ts.load_mutex);
    ts.load_cv.wait(lock, [&] {
      return ts.loaded_count == static_cast<int>(mc.tp_degree);
    });
  }
  st.load_ms = load_timer.ms();
  std::fprintf(stderr,
               "[model loader] device models loaded in %.2f ms (%.2f s)\n",
               st.load_ms, st.load_ms / 1000.0);

  std::exception_ptr err;
  for (auto& m : out.models_) {
    if (auto e = m->error()) {
      err = e;
      break;
    }
  }
  if (err) {
    {
      std::lock_guard<std::mutex> lock(ts.work_mutex);
      ts.shutdown = true;
    }
    ts.work_cv.notify_all();
    for (auto& t : out.workers_)
      if (t.joinable()) t.join();
    std::rethrow_exception(err);
  }

  // Broker the cross-rank pointer exchange: every rank's Trunk io buffer and
  // allreduce flag word are allocated and stable now, so collect the
  // pointers and broadcast a SetPeerIo work item (each Model builds its
  // device-side pointer tables + pipe-path scratch from them). This must
  // happen before the first ForwardRun. (Single-process: device pointers are
  // plain shareable values; the main thread reads them with no device call.)
  if (mc.tp_degree > 1) {
    gpu::WorkItem w;
    w.kind = gpu::WorkItem::Kind::SetPeerIo;
    w.peer_io.resize(mc.tp_degree);
    w.peer_flags.resize(mc.tp_degree);
    for (uint32_t r = 0; r < mc.tp_degree; ++r) {
      w.peer_io[r] = out.models_[r]->io_ptr();
      w.peer_flags[r] = out.models_[r]->ar_flag_ptr();
    }
    out.submit_work(w);
  }

  out.sampler_ = std::make_unique<Sampler>(sc);
  return out;
}

Engine init_engine(const std::filesystem::path& model_file,
                   const ModelConfig& mc, const SamplingConfig& sc,  LoadStats* out_stats) {
  LoadStats dummy;
  try {
    return init_engine_impl(model_file, mc, sc, out_stats ? *out_stats : dummy);
  } catch (const json::exception& e) {
    fail(std::string("descriptor: ") + e.what());
  }
}

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void print_stats(const char* phase, size_t tokens, double seconds) {
  if (tokens == 0) {
    std::fprintf(stderr, "[%s: 0 tokens in %.3f s]\n", phase, seconds);
    return;
  }
  std::fprintf(stderr,
               "[%s: %zu tokens in %.3f s — %.2f ms/token, %.1f tok/s]\n",
               phase, tokens, seconds, seconds * 1e3 / double(tokens),
               double(tokens) / seconds);
}

size_t utf8_complete_len(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    const auto b = uint8_t(s[i]);
    const size_t len =
        b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : 4;
    if (i + len > s.size()) break;
    i += len;
  }
  return i;
}

}  // namespace

std::vector<int32_t> Engine::encode(const std::string& text) const {
  return tokenizer_.encode(text);
}

std::string Engine::decode(const std::vector<int32_t>& ids) const {
  return tokenizer_.decode(ids);
}

std::vector<float> Engine::forward_logits(std::span<const int32_t> tokens) {
  submit_work(gpu::WorkItem{gpu::WorkItem::Kind::ForwardLogits,
                            std::vector<int32_t>(tokens.begin(), tokens.end())});
  return models_[0]->last_logits();
}

std::vector<float> Engine::forward_logits_all(std::span<const int32_t> tokens) {
  submit_work(gpu::WorkItem{gpu::WorkItem::Kind::ForwardLogitsAll,
                            std::vector<int32_t>(tokens.begin(), tokens.end())});
  return models_[0]->last_logits();
}

int32_t Engine::forward(std::span<const int32_t> tokens) {
  const auto logits = forward_logits(tokens);
  return sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
}

void Engine::reset() {
  submit_work(gpu::WorkItem{gpu::WorkItem::Kind::Reset, {}});
}

Engine::GenerateResult Engine::generate(
    const std::string& prompt, const GenerateOptions& opts,
    std::function<void(const std::string&)> on_token) {
  GenerateResult result;
  reset();

  auto ids = tokenizer_.encode(prompt);
  ids.insert(ids.begin(), int32_t(tokenizer_.bos_token_id()));
  if (ids.size() >= opts.max_seq_len)
    throw std::runtime_error("generate: prompt too long for max_seq_len");
  result.prompt_tokens = ids.size();

  const auto is_eos = [&](int32_t id) {
    const auto& eos = tokenizer_.eos_token_ids();
    return std::find(eos.begin(), eos.end(), id) != eos.end();
  };

  const auto prefill_t0 = Clock::now();
  int32_t t = -1;
  for (size_t off = 0; off < ids.size(); off += opts.max_prefill_chunk) {
    const size_t n = std::min<size_t>(opts.max_prefill_chunk, ids.size() - off);
    const bool last = off + n >= ids.size();
    if (last) {
      const auto logits = forward_logits(std::span<const int32_t>(ids.data() + off, n));
      t = sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
    } else {
      submit_work(gpu::WorkItem{gpu::WorkItem::Kind::ForwardRun,
                                std::vector<int32_t>(ids.data() + off,
                                                     ids.data() + off + n)});
    }
  }
  result.prefill_seconds = elapsed_s(prefill_t0);
  print_stats("prefill", ids.size(), result.prefill_seconds);

  const auto decode_t0 = Clock::now();
  size_t shown = 0;
  std::vector<int32_t> gen;
  for (int step = 0; step < opts.max_new_tokens; ++step) {
    if (is_eos(t)) break;
    gen.push_back(t);
    if (on_token) {
      const std::string full = tokenizer_.decode(gen);
      const size_t complete = utf8_complete_len(full);
      if (complete > shown) {
        on_token(full.substr(shown, complete - shown));
        shown = complete;
      }
    }
    const auto logits = forward_logits(std::span<const int32_t>(&t, 1));
    t = sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
  }
  if (on_token && !gen.empty()) {
    const std::string full = tokenizer_.decode(gen);
    const size_t complete = utf8_complete_len(full);
    if (complete > shown) on_token(full.substr(shown, complete - shown));
  }
  result.text = tokenizer_.decode(gen);
  result.generated_tokens = gen.size();
  result.decode_seconds = elapsed_s(decode_t0);
  print_stats("decode", gen.size(), result.decode_seconds);
  return result;
}

}  // namespace engine

}  // namespace alice
