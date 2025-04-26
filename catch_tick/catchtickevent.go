package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strconv"

	dem "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

type KillInfo struct {
	GroupID    int    `json:"group_id"`
	Tick       int    `json:"tick"`
	KillerName string `json:"killer_name"`
	VictimName string `json:"victim_name"`
	WeaponName string `json:"weapon_name"`
	DemoID     int    `json:"demo_id"`
}

type TickData struct {
	GroupID int
	Tick    int
	ViewX   float32
	ViewY   float32
	Fired   int
	DemoID  int
}

type KillTickProgress struct {
	LastProcessedIndex int `json:"last_processed_index"`
}

const progressKillTickFile = "logs/progress_kills_to_tick.json"

func loadKillTickProgress() int {
	data, err := ioutil.ReadFile(progressKillTickFile)
	if err != nil {
		return -1
	}
	var prog KillTickProgress
	_ = json.Unmarshal(data, &prog)
	return prog.LastProcessedIndex
}

func saveKillTickProgress(index int) {
	prog := KillTickProgress{LastProcessedIndex: index}
	data, _ := json.MarshalIndent(prog, "", "  ")
	_ = ioutil.WriteFile(progressKillTickFile, data, 0644)
}

func parseDemoTicks(demoID int, kills []KillInfo, buffer map[int]map[int]TickData, warned map[int]bool) {
	fmt.Printf("📁 開始處理 DEM：game%d.dem\n", demoID)
	path := fmt.Sprintf("REPLY/game%d.dem", demoID)
	f, err := os.Open(path)
	if err != nil {
		log.Printf("跳過 %s，無法開啟: %v", path, err)
		return
	}
	defer f.Close()
	p := dem.NewParser(f)
	defer p.Close()

	for _, kill := range kills {
		buffer[kill.GroupID] = make(map[int]TickData)

		p.RegisterEventHandler(func(e events.FrameDone) {
			curTick := p.CurrentFrame()
			if curTick < kill.Tick-63 || curTick > kill.Tick {
				return
			}

			var foundKiller, foundVictim bool
			var vx, vy float32
			for _, pl := range p.GameState().Participants().Playing() {
				if pl.Name == kill.KillerName {
					vx, vy = pl.ViewDirectionX(), pl.ViewDirectionY()
					foundKiller = true
				}
				if pl.Name == kill.VictimName {
					foundVictim = true
				}
			}
			if !foundKiller || !foundVictim {
				if !warned[kill.GroupID] {
					log.Printf("找不到玩家，GroupID %d (game%d.dem)", kill.GroupID, demoID)
					warned[kill.GroupID] = true
				}
				return
			}

			buffer[kill.GroupID][curTick] = TickData{
				GroupID: kill.GroupID,
				Tick:    curTick,
				ViewX:   vx,
				ViewY:   vy,
				Fired:   0,
				DemoID:  kill.DemoID,
			}
		})

		p.RegisterEventHandler(func(e events.WeaponFire) {
			curTick := p.CurrentFrame()
			if e.Shooter != nil && e.Shooter.Name == kill.KillerName {
				if d, ok := buffer[kill.GroupID][curTick]; ok {
					d.Fired = 1
					buffer[kill.GroupID][curTick] = d
				}
			}
		})
	}

	if err := p.ParseToEnd(); err != nil {
		log.Printf("解析 demo%d 失敗: %v", demoID, err)
	}
}

func writeCSV(path string, buffer map[int]map[int]TickData) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"GroupID", "Tick", "ViewX", "ViewY", "Fired", "DemoID"})

	for groupID, ticks := range buffer {
		if len(ticks) < 64 {
			continue
		}
		fmt.Printf("🔍 處理 GroupID: %d（tick 數：%d）\n", groupID, len(ticks))
		var keys []int
		for tick := range ticks {
			keys = append(keys, tick)
		}
		sort.Ints(keys)
		for _, tick := range keys {
			d := ticks[tick]
			record := []string{
				strconv.Itoa(d.GroupID),
				strconv.Itoa(d.Tick),
				fmt.Sprintf("%.2f", d.ViewX),
				fmt.Sprintf("%.2f", d.ViewY),
				strconv.Itoa(d.Fired),
				strconv.Itoa(d.DemoID),
			}
			w.Write(record)
		}
	}
	return nil
}

func main() {
	outputIndex := os.Getenv("OUTPUT_INDEX")
	if outputIndex == "" {
		log.Fatal("請設置 OUTPUT_INDEX")
	}
	idx, _ := strconv.Atoi(outputIndex)

	reprocess := os.Getenv("REPROCESS_KILL_INDEX")
	puredataPath := fmt.Sprintf("output/puredata/puredata%s.csv", outputIndex)
	if reprocess == "" && idx <= loadKillTickProgress() {
		if _, err := os.Stat(puredataPath); err == nil {
			log.Printf("%s 已處理過，跳過", puredataPath)
			return
		}
		log.Printf("⚠️ 雖然 progress 記錄已處理，但 %s 不存在，將重新處理", puredataPath)
	}

	killPath := fmt.Sprintf("output/kills/kills%s.json", outputIndex)
	data, err := ioutil.ReadFile(killPath)
	if err != nil {
		log.Fatalf("無法開啟 %s: %v", killPath, err)
	}
	var kills []KillInfo
	_ = json.Unmarshal(data, &kills)

	demoMap := make(map[int][]KillInfo)
	for _, k := range kills {
		demoMap[k.DemoID] = append(demoMap[k.DemoID], k)
	}

	tickBuf := make(map[int]map[int]TickData)
	warned := make(map[int]bool)
	for demoID, ks := range demoMap {
		parseDemoTicks(demoID, ks, tickBuf, warned)
	}

	outPath := fmt.Sprintf("output/puredata/puredata%s.csv", outputIndex)
	if err := writeCSV(outPath, tickBuf); err != nil {
		log.Panic("寫入 CSV 失敗: ", err)
	}

	saveKillTickProgress(idx)
	fmt.Printf("✅ 完成輸出 %s\n", outPath)
}
