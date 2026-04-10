package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// cpuStat holds raw values from /proc/stat for a single CPU line
type cpuStat struct {
	user    uint64
	nice    uint64
	system  uint64
	idle    uint64
	iowait  uint64
	irq     uint64
	softirq uint64
	steal   uint64
}

func (c cpuStat) total() uint64 {
	return c.user + c.nice + c.system + c.idle + c.iowait + c.irq + c.softirq + c.steal
}

func (c cpuStat) idleTotal() uint64 {
	return c.idle + c.iowait
}

// raspCollector implements prometheus.Collector, reading metrics on each scrape
type raspCollector struct {
	mu      sync.Mutex
	prevCPU map[string]cpuStat

	descCPUUsage     *prometheus.Desc
	descMemUsedBytes *prometheus.Desc
	descMemUsedMB    *prometheus.Desc
	descMemUsedPct   *prometheus.Desc
	descTemp         *prometheus.Desc
}

func newRaspCollector() *raspCollector {
	labels := []string{"cpu"}
	return &raspCollector{
		prevCPU: make(map[string]cpuStat),

		descCPUUsage: prometheus.NewDesc(
			"rasp_cpu_usage_percent",
			"CPU usage percentage. Label 'cpu': 'total' or core index (0, 1, ...)",
			labels, nil,
		),
		descMemUsedBytes: prometheus.NewDesc(
			"rasp_memory_used_bytes",
			"Memory used in bytes (MemTotal - MemAvailable)",
			nil, nil,
		),
		descMemUsedMB: prometheus.NewDesc(
			"rasp_memory_used_mb",
			"Memory used in megabytes (MemTotal - MemAvailable)",
			nil, nil,
		),
		descMemUsedPct: prometheus.NewDesc(
			"rasp_memory_used_percent",
			"Memory usage as a percentage of total",
			nil, nil,
		),
		descTemp: prometheus.NewDesc(
			"rasp_cpu_temperature_celsius",
			"CPU temperature in Celsius from thermal_zone0",
			nil, nil,
		),
	}
}

func (r *raspCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- r.descCPUUsage
	ch <- r.descMemUsedBytes
	ch <- r.descMemUsedMB
	ch <- r.descMemUsedPct
	ch <- r.descTemp
}

func (r *raspCollector) Collect(ch chan<- prometheus.Metric) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.collectCPU(ch)
	r.collectMemory(ch)
	r.collectTemperature(ch)
}

// collectCPU reads /proc/stat and computes usage since last call
func (r *raspCollector) collectCPU(ch chan<- prometheus.Metric) {
	f, err := os.Open("/proc/stat")
	if err != nil {
		log.Printf("cpu: cannot open /proc/stat: %v", err)
		return
	}
	defer f.Close()

	current := make(map[string]cpuStat)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "cpu") {
			break // cpu lines are always at the top; stop early
		}
		fields := strings.Fields(line)
		if len(fields) < 9 {
			continue
		}
		name := fields[0]
		var s cpuStat
		vals := []*uint64{&s.user, &s.nice, &s.system, &s.idle, &s.iowait, &s.irq, &s.softirq, &s.steal}
		for i, ptr := range vals {
			v, err := strconv.ParseUint(fields[i+1], 10, 64)
			if err != nil {
				continue
			}
			*ptr = v
		}
		current[name] = s
	}

	for name, curr := range current {
		prev, ok := r.prevCPU[name]
		r.prevCPU[name] = curr
		if !ok {
			continue // first read — no delta yet
		}

		totalDiff := float64(curr.total() - prev.total())
		idleDiff := float64(curr.idleTotal() - prev.idleTotal())

		var pct float64
		if totalDiff > 0 {
			pct = (1.0 - idleDiff/totalDiff) * 100.0
		}

		label := name
		if name == "cpu" {
			label = "total"
		} else {
			// "cpu0" → "0"
			label = strings.TrimPrefix(name, "cpu")
		}

		ch <- prometheus.MustNewConstMetric(r.descCPUUsage, prometheus.GaugeValue, pct, label)
	}
}

// collectMemory reads /proc/meminfo
func (r *raspCollector) collectMemory(ch chan<- prometheus.Metric) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		log.Printf("mem: cannot open /proc/meminfo: %v", err)
		return
	}
	defer f.Close()

	info := make(map[string]uint64, 8)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		key := strings.TrimSuffix(fields[0], ":")
		val, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		info[key] = val
		// MemTotal and MemAvailable appear early; stop after we have both
		if _, ok1 := info["MemTotal"]; ok1 {
			if _, ok2 := info["MemAvailable"]; ok2 {
				break
			}
		}
	}

	total, ok1 := info["MemTotal"]
	available, ok2 := info["MemAvailable"]
	if !ok1 || !ok2 || total == 0 {
		log.Printf("mem: missing MemTotal or MemAvailable")
		return
	}

	usedKB := total - available
	usedBytes := float64(usedKB * 1024)
	usedMB := float64(usedKB) / 1024.0
	pct := float64(usedKB) / float64(total) * 100.0

	ch <- prometheus.MustNewConstMetric(r.descMemUsedBytes, prometheus.GaugeValue, usedBytes)
	ch <- prometheus.MustNewConstMetric(r.descMemUsedMB, prometheus.GaugeValue, usedMB)
	ch <- prometheus.MustNewConstMetric(r.descMemUsedPct, prometheus.GaugeValue, pct)
}

// collectTemperature reads /sys/class/thermal/thermal_zone0/temp
func (r *raspCollector) collectTemperature(ch chan<- prometheus.Metric) {
	data, err := os.ReadFile("/sys/class/thermal/thermal_zone0/temp")
	if err != nil {
		log.Printf("temp: cannot read thermal_zone0: %v", err)
		return
	}

	milliCelsius, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
	if err != nil {
		log.Printf("temp: cannot parse value: %v", err)
		return
	}

	ch <- prometheus.MustNewConstMetric(r.descTemp, prometheus.GaugeValue, milliCelsius/1000.0)
}

func main() {
	port := flag.Int("port", 9100, "Port to expose /metrics on")
	flag.Parse()

	collector := newRaspCollector()

	// Warm-up read so the first real scrape has a CPU delta
	func() {
		collector.mu.Lock()
		defer collector.mu.Unlock()
		// discard channel — we just want prevCPU populated
		collector.collectCPU(make(chan prometheus.Metric, 64))
	}()
	time.Sleep(time.Second)

	registry := prometheus.NewRegistry()
	registry.MustRegister(collector)

	http.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{
		EnableOpenMetrics: false,
	}))
	http.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, `<html><body><a href="/metrics">Metrics</a></body></html>`)
	})

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("rasp_monitor listening on %s — exposing /metrics", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
