// ✅ 按 GroupID + DemoFilename 進度追蹤版 catchtickevent.go（含 IsTarget、WeaponName，並追加 CSV）

package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	dem "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/common"
	events "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

// KillInfo 對應 kills.json 的結構
// WeaponName 與 IsTarget 從這裡直接帶入 TickData
type KillInfo struct {
	GroupID      int    `json:"group_id"`
	Tick         int    `json:"tick"`
	KillerName   string `json:"killer_name"`
	VictimName   string `json:"victim_name"`
	WeaponName   string `json:"weapon_name"`
	DemoFilename string `json:"demo_filename"`
	DemoTime     string `json:"demo_time"`
	IsTarget     int    `json:"is_target"`
}

// TickData 保存 64 tick 內每帧的玩家狀態、標記與附加資訊
type TickData struct {
	GroupID                            int
	Tick                               int
	ViewX, ViewY                       float32
	Fired                              int
	IsScoped, IsDucking, IsWalking     int
	IsTarget                           int // 新增：來自 killInfo
	PosX, PosY, PosZ                   float32
	VelX, VelY, VelZ                   float32
	WeaponName                         string // 直接使用 killInfo.WeaponName
	VictimPosX, VictimPosY, VictimPosZ float32
	VictimVelX, VictimVelY, VictimVelZ float32
	DemoTime                           string // demo 日期
}

// GroupProgress 用於記錄已處理到的最大 GroupID
type GroupProgress struct {
	LastProcessedGroupID int `json:"last_processed_group_id"`
}

const (
	// 相對於 catch_tick 執行目錄
	progressFile = "../logs/progress_group.json"
	killJSON     = "../output/kills/kills.json"
	outCSV       = "../output/puredata/puredata.csv"
)

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// loadGroupProgress 讀取上次處理的最大 GroupID
func loadGroupProgress() int {
	data, err := ioutil.ReadFile(progressFile)
	if err != nil {
		return -1
	}
	var prog GroupProgress
	_ = json.Unmarshal(data, &prog)
	return prog.LastProcessedGroupID
}

// saveGroupProgress 寫入最新處理的最大 GroupID
func saveGroupProgress(id int) {
	prog := GroupProgress{LastProcessedGroupID: id}
	data, _ := json.MarshalIndent(prog, "", "  ")
	_ = os.WriteFile(progressFile, data, 0644)
}

// parseDemoTicks 收集單個 DEM (路徑來自 DemoFilename) 的 kills
func parseDemoTicks(demoPath string, kills []KillInfo, buffer map[int]map[int]TickData, warned map[int]bool) {
	f, err := os.Open(demoPath)
	if err != nil {
		log.Printf("跳過 %s: %v", demoPath, err)
		return
	}
	defer f.Close()
	p := dem.NewParser(f)
	defer p.Close()

	for _, kill := range kills {
		buffer[kill.GroupID] = make(map[int]TickData)

		p.RegisterEventHandler(func(e events.FrameDone) {
			cur := p.CurrentFrame()
			if cur < kill.Tick-63 || cur > kill.Tick {
				return
			}
			var killer, victim *common.Player
			for _, pl := range p.GameState().Participants().Playing() {
				if pl.Name == kill.KillerName {
					killer = pl
				}
				if pl.Name == kill.VictimName {
					victim = pl
				}
			}
			if killer == nil || victim == nil {
				if !warned[kill.GroupID] {
					log.Printf("找不到玩家 GroupID %d", kill.GroupID)
					warned[kill.GroupID] = true
				}
				return
			}
			pos, vel := killer.Position(), killer.Velocity()
			vpos, vvel := victim.Position(), victim.Velocity()
			buffer[kill.GroupID][cur] = TickData{
				GroupID:   kill.GroupID,
				Tick:      cur,
				ViewX:     killer.ViewDirectionX(),
				ViewY:     killer.ViewDirectionY(),
				Fired:     0,
				IsScoped:  btoi(killer.IsScoped()),
				IsDucking: btoi(killer.IsDucking()),
				IsWalking: btoi(killer.IsWalking()),
				IsTarget:  kill.IsTarget,
				PosX:      float32(pos.X), PosY: float32(pos.Y), PosZ: float32(pos.Z),
				VelX: float32(vel.X), VelY: float32(vel.Y), VelZ: float32(vel.Z),
				WeaponName: kill.WeaponName,
				VictimPosX: float32(vpos.X), VictimPosY: float32(vpos.Y), VictimPosZ: float32(vpos.Z),
				VictimVelX: float32(vvel.X), VictimVelY: float32(vvel.Y), VictimVelZ: float32(vvel.Z),
				DemoTime: kill.DemoTime,
			}
		})

		p.RegisterEventHandler(func(e events.WeaponFire) {
			cur := p.CurrentFrame()
			if e.Shooter != nil && e.Shooter.Name == kill.KillerName {
				if d, ok := buffer[kill.GroupID][cur]; ok {
					d.Fired = 1
					buffer[kill.GroupID][cur] = d
				}
			}
		})
	}
	_ = p.ParseToEnd()
}

// writeCSV 輸出純資料，只保留滿 64 tick 的 Group，並以追加模式寫入
func writeCSV(path string, buffer map[int]map[int]TickData) error {
	// 確保目錄存在
	dir := filepath.Dir(path)
	os.MkdirAll(dir, os.ModePerm)

	// 判断是否需要写表头
	needHeader := false
	if fi, err := os.Stat(path); os.IsNotExist(err) || fi.Size() == 0 {
		needHeader = true
	}

	// 以 append 模式打开
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	// 写表头
	if needHeader {
		w.Write([]string{
			"GroupID", "Tick", "ViewX", "ViewY", "Fired",
			"IsScoped", "IsDucking", "IsWalking", "IsTarget",
			"PosX", "PosY", "PosZ", "VelX", "VelY", "VelZ",
			"WeaponName", "VictimPosX", "VictimPosY", "VictimPosZ",
			"VictimVelX", "VictimVelY", "VictimVelZ", "DemoTime",
		})
	}

	// 输出数据行
	for _, ticks := range buffer {
		if len(ticks) < 64 {
			continue
		}
		keys := make([]int, 0, len(ticks))
		for t := range ticks {
			keys = append(keys, t)
		}
		sort.Ints(keys)
		for _, t := range keys {
			d := ticks[t]
			rec := []string{
				strconv.Itoa(d.GroupID), strconv.Itoa(d.Tick),
				fmt.Sprintf("%.2f", d.ViewX), fmt.Sprintf("%.2f", d.ViewY), strconv.Itoa(d.Fired),
				strconv.Itoa(d.IsScoped), strconv.Itoa(d.IsDucking), strconv.Itoa(d.IsWalking),
				strconv.Itoa(d.IsTarget),
				fmt.Sprintf("%.2f", d.PosX), fmt.Sprintf("%.2f", d.PosY), fmt.Sprintf("%.2f", d.PosZ),
				fmt.Sprintf("%.2f", d.VelX), fmt.Sprintf("%.2f", d.VelY), fmt.Sprintf("%.2f", d.VelZ),
				d.WeaponName,
				fmt.Sprintf("%.2f", d.VictimPosX), fmt.Sprintf("%.2f", d.VictimPosY), fmt.Sprintf("%.2f", d.VictimPosZ),
				fmt.Sprintf("%.2f", d.VictimVelX), fmt.Sprintf("%.2f", d.VictimVelY), fmt.Sprintf("%.2f", d.VictimVelZ),
				d.DemoTime,
			}
			w.Write(rec)
		}
	}
	return nil
}

func main() {
	lastGroup := loadGroupProgress()
	log.Printf("從 GroupID %d 開始處理", lastGroup+1)

	data, err := ioutil.ReadFile(killJSON)
	if err != nil {
		log.Fatalf("無法讀取 %s: %v", killJSON, err)
	}
	var kills []KillInfo
	_ = json.Unmarshal(data, &kills)

	var newKills []KillInfo
	for _, k := range kills {
		if k.GroupID > lastGroup {
			newKills = append(newKills, k)
		}
	}
	if len(newKills) == 0 {
		log.Println("沒有新 Group 要處理，退出。🎉")
		return
	}

	demoMap := make(map[string][]KillInfo)
	for _, k := range newKills {
		demoMap[k.DemoFilename] = append(demoMap[k.DemoFilename], k)
	}

	tickBuf := make(map[int]map[int]TickData)
	warned := make(map[int]bool)
	for fname, ks := range demoMap {
		demoPath := filepath.Join("../..", "DEM", fname)
		parseDemoTicks(demoPath, ks, tickBuf, warned)
	}

	if err := writeCSV(outCSV, tickBuf); err != nil {
		log.Fatalf("寫入 CSV 失敗: %v", err)
	}

	maxGroup := lastGroup
	for _, k := range newKills {
		if k.GroupID > maxGroup {
			maxGroup = k.GroupID
		}
	}
	saveGroupProgress(maxGroup)
	fmt.Printf("✅ 完成輸出 %s，已處理到 GroupID %d\n", outCSV, maxGroup)
}
