// Read-only ROCm SMI telemetry. No clocks, power limits or profiles are set.
#include <rocm_smi/rocm_smi.h>
#include <chrono>
#include <signal.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

using Clock=std::chrono::steady_clock;

int main(int argc,char** argv) try {
  if(argc<4 || argc>5) throw std::runtime_error("usage: clock-monitor PID OUT PCI [MAX_SAMPLES=0]");
  const int pid=std::stoi(argv[1]), maximum=argc==5?std::stoi(argv[4]):0;
  const std::filesystem::path out=argv[2];
  std::string pci=argv[3];
  if(pid<=0 || maximum<0) throw std::runtime_error("invalid PID/sample limit");
  if(rsmi_init(0)!=RSMI_STATUS_SUCCESS) throw std::runtime_error("rsmi_init failed");
  { std::ofstream ready(out/"clock-ready"); ready<<"SMI initialized\n"; }
  while(pci=="auto" && kill(pid,0)==0 && !std::filesystem::exists(out/"clock-stop")) {
    std::ifstream meta(out/"device.txt"); std::string token;
    while(meta>>token) if(token.starts_with("PCI=")) { pci=token.substr(4); break; }
    if(pci=="auto") std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if(pci=="auto") { rsmi_shut_down(); return 0; }
  uint32_t devices=0; rsmi_num_monitor_devices(&devices);
  uint32_t device=devices;
  for(uint32_t i=0;i<devices;++i) {
    uint64_t bdf=0;
    if(rsmi_dev_pci_id_get(i,&bdf)!=RSMI_STATUS_SUCCESS) continue;
    char name[32];
    std::snprintf(name,sizeof(name),"%04x:%02x:%02x.%x",unsigned(bdf>>32),
                  unsigned((bdf>>8)&255),unsigned((bdf>>3)&31),unsigned(bdf&7));
    if(pci==name) { device=i; break; }
  }
  if(device==devices) throw std::runtime_error("PCI device not found in ROCm SMI");
  std::ofstream f(out/"clocks.csv");
  f<<"unix_s,stage,pattern,bytes,offset_bytes,metrics_status,metrics_format,metrics_content,"
      "firmware_timestamp,current_gfx_MHz,average_gfx_MHz,current_soc_MHz,average_soc_MHz,"
      "current_mem_MHz,average_mem_MHz,reported_fclk_Hz,fclk_status,fclk_deep_sleep_support,"
      "socket_W,edge_C,junction_C,memory_C,gfx_busy_percent,umc_busy_percent,"
      "throttle_status,indep_throttle_status,query_ms\n"<<std::setprecision(17);
  for(int sample=0;kill(pid,0)==0 && !std::filesystem::exists(out/"clock-stop") && (!maximum || sample<maximum);++sample) {
    const auto start=Clock::now();
    std::string phase="initializing,NA,0,0";
    std::ifstream status(out/"status.txt");
    std::string line; if(std::getline(status,line) && !line.empty()) phase=line;
    rsmi_gpu_metrics_t m{};
    const auto metrics=rsmi_dev_gpu_metrics_info_get(device,&m);
    rsmi_frequencies_t frequencies{};
    const auto fabric=rsmi_dev_gpu_clk_freq_get(device,RSMI_CLK_TYPE_DF,&frequencies);
    const bool valid=fabric==RSMI_STATUS_SUCCESS && frequencies.current<frequencies.num_supported;
    const double query_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    const double unix_s=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    f<<unix_s<<','<<phase<<','<<int(metrics)<<','<<int(m.common_header.format_revision)<<','
     <<int(m.common_header.content_revision)<<','<<m.firmware_timestamp<<','
     <<m.current_gfxclk<<','<<m.average_gfxclk_frequency<<','<<m.current_socclk<<','
     <<m.average_socclk_frequency<<','<<m.current_uclk<<','<<m.average_uclk_frequency<<',';
    if(valid) f<<frequencies.frequency[frequencies.current]; else f<<"NA";
    f<<','<<int(fabric)<<','<<frequencies.has_deep_sleep<<','<<m.average_socket_power<<','
     <<m.temperature_edge<<','<<m.temperature_hotspot<<','<<m.temperature_mem<<','
     <<m.average_gfx_activity<<','<<m.average_umc_activity<<','<<m.throttle_status<<','
     <<m.indep_throttle_status<<','<<query_ms<<'\n'; f.flush();
    std::this_thread::sleep_until(start+std::chrono::milliseconds(100));
  }
  rsmi_shut_down();
  return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
