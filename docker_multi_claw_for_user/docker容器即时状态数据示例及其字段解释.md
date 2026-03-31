# 数据示例
```json
{
    "id": "ddc52d18940e0e6d9b2bbb43df6b3334fc259eb3181121fd3e85496ba896d2cf",
    "name": "/copaw-user-user1",
    "os_type": "linux",
    "read": "2026-03-27T03:17:20.258616305Z",
    "cpu_stats": {
        "cpu_usage": {
            "total_usage": 17530096000,
            "usage_in_kernelmode": 4924767000,
            "usage_in_usermode": 12605328000
        },
        "system_cpu_usage": 96921790000000,
        "online_cpus": 16,
        "throttling_data": {
            "periods": 1325,
            "throttled_periods": 325,
            "throttled_time": 80149150000
        }
    },
    "memory_stats": {
        "usage": 710574080,
        "stats": {
            "active_anon": 139264,
            "active_file": 1552384,
            "anon": 651124736,
            "anon_thp": 0,
            "file": 7643136,
            "file_dirty": 28672,
            "file_mapped": 3330048,
            "file_writeback": 0,
            "inactive_anon": 651431936,
            "inactive_file": 5636096,
            "kernel_stack": 2981888,
            "pgactivate": 4568,
            "pgdeactivate": 1,
            "pgfault": 298854,
            "pglazyfree": 0,
            "pglazyfreed": 0,
            "pgmajfault": 826,
            "pgrefill": 0,
            "pgscan": 0,
            "pgsteal": 0,
            "shmem": 454656,
            "slab": 41008648,
            "slab_reclaimable": 35225744,
            "slab_unreclaimable": 5782904,
            "sock": 0,
            "thp_collapse_alloc": 0,
            "thp_fault_alloc": 0,
            "unevictable": 0,
            "workingset_activate": 0,
            "workingset_nodereclaim": 0,
            "workingset_refault": 0
        },
        "limit": 1073741824
    },
    "networks": {
        "eth0": {
            "rx_bytes": 97099,
            "rx_packets": 287,
            "rx_errors": 0,
            "rx_dropped": 0,
            "tx_bytes": 441815,
            "tx_packets": 262,
            "tx_errors": 0,
            "tx_dropped": 0
        }
    },
    "pids_stats": {
        "current": 182,
        "limit": 18446744073709551615
    },
    "blkio_stats": {
        "io_service_bytes_recursive": [
            {
                "major": 7,
                "minor": 1,
                "op": "read",
                "value": 16384
            },
            {
                "major": 7,
                "minor": 1,
                "op": "write",
                "value": 0
            },
            {
                "major": 8,
                "minor": 64,
                "op": "read",
                "value": 226566144
            },
            {
                "major": 8,
                "minor": 64,
                "op": "write",
                "value": 811008
            }
        ],
        "io_serviced_recursive": None,
        "io_queue_recursive": None,
        "io_service_time_recursive": None,
        "io_wait_time_recursive": None,
        "io_merged_recursive": None,
        "io_time_recursive": None,
        "sectors_recursive": None
    },
    "num_procs": 0,
    "storage_stats": {},
    "preread": "2026-03-27T03:17:19.254687957Z",
    "precpu_stats": {
        "cpu_usage": {
            "total_usage": 17524456000,
            "usage_in_kernelmode": 4924767000,
            "usage_in_usermode": 12599688000
        },
        "system_cpu_usage": 96905750000000,
        "online_cpus": 16,
        "throttling_data": {
            "periods": 1317,
            "throttled_periods": 325,
            "throttled_time": 80149150000
        }
    }
}
```

# 指标解释及单位情况

我直接对照你这份 **Docker 容器实时统计**，把**每个字段的标准单位**一次性给全，方便你写监控、计算使用率。

# 一、时间相关
- `read` / `preread`：**ISO8601 时间字符串**
- `throttling_time`：**纳秒 nanoseconds (ns)**
- 所有 `usage` 类 CPU 累计值：**纳秒 ns**

# 二、CPU 相关单位
- `cpu_usage.total_usage`：**纳秒 ns**（容器自启动以来总 CPU 耗时）
- `usage_in_kernelmode`：**纳秒 ns**（内核态）
- `usage_in_usermode`：**纳秒 ns**（用户态）
- `system_cpu_usage`：**纳秒 ns**（宿主机总系统 CPU 时间）
- `online_cpus`：**个**（逻辑 CPU 核心数）
- `throttling_data.periods` / `throttled_periods`：**次**
- `throttling_time`：**纳秒 ns**

# 三、内存相关单位
- `usage`：**字节 Byte**
- `limit`：**字节 Byte**
- `stats` 下所有内存指标（`anon`/`file`/`slab`/`shmem`/`kernel_stack` 等）：**字节 Byte**
- `pgfault` / `pgmajfault` 等：**次**（缺页中断次数）

# 四、网络（networks）
- `rx_bytes` / `tx_bytes`：**字节 Byte**
- `rx_packets` / `tx_packets`：**个**（数据包数）
- `rx_errors` / `tx_errors` / `dropped`：**个**

# 五、进程（pids_stats）
- `current`：**个**
- `limit`：**个**

# 六、块设备 I/O（blkio_stats）
- `io_service_bytes_recursive.value`：**字节 Byte**
- 其余 `io_serviced` 等若有值：**次**

---

# 给你几个常用换算（方便你做面板）
- **Bytes → MB**：÷ 1024 ÷ 1024
- **Bytes → GB**：÷ 1024 ÷ 1024 ÷ 1024
- **ns → 秒**：÷ 1e9
- **ns → 毫秒**：÷ 1e6

---

# 你当前容器的直观读数
- 内存使用：**~677.66 MB**
- 内存上限：**1 GB**
- CPU 被限流时间：**80.15 秒**
- 进程数：**182**

如果你需要，我可以直接帮你写一段**Python 代码，把这段 JSON 转成人类可读的监控面板**（使用率%、内存MB、CPU 负载等）。