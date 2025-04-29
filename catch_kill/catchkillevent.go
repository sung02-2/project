package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	dem "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

type KillInfo struct {
	GroupID    int    `json:"group_id"`
	Tick       int    `json:"tick"`
	KillerName string `json:"killer_name"`
	VictimName string `json:"victim_name"`
	WeaponName string `json:"weapon_name"`
	DemoName   string `json:"demo_filename"`
	DemoTime   string `json:"demo_time"`
	IsTarget   int    `json:"is_target"` // 0: 非目標玩家，1: 目標玩家
}

func loadProcessedDEMSet(logPath string) map[string]bool {
	processed := map[string]bool{}
	data, err := os.ReadFile(logPath)
	if err != nil {
		return processed
	}
	for _, line := range strings.Split(string(data), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			processed[line] = true
		}
	}
	return processed
}

func appendToLog(logPath, filename string) {
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("⚠️ 無法寫入 killlog：", err)
		return
	}
	defer f.Close()
	f.WriteString(filename + "\n")
}

func processDemo(demPath, filename, demoTime string, targets map[string]bool, allKills *[]KillInfo, groupCounter *int) int {
	// 1. 收集所有擊殺事件
	var localKills []KillInfo

	f, err := os.Open(demPath)
	if err != nil {
		log.Println("❌ 無法開啟 DEM:", demPath)
		return 0
	}
	defer f.Close()

	p := dem.NewParser(f)
	defer p.Close()

	p.RegisterEventHandler(func(e events.Kill) {
		if e.Killer == nil || e.Victim == nil || e.Killer.ActiveWeapon() == nil {
			return
		}
		isTarget := 0
		if targets[strings.ToLower(e.Killer.Name)] {
			isTarget = 1
		}
		localKills = append(localKills, KillInfo{
			Tick:       p.CurrentFrame(),
			KillerName: e.Killer.Name,
			VictimName: e.Victim.Name,
			WeaponName: e.Killer.ActiveWeapon().Type.String(),
			DemoName:   filename,
			DemoTime:   demoTime,
			IsTarget:   isTarget,
		})
	})
	p.ParseToEnd()

	// 2. 過濾：對每筆目標玩家擊殺，附加一筆後續非目標（除非是最後一筆）
	var filtered []KillInfo
	for i, k := range localKills {
		if k.IsTarget == 1 {
			k.GroupID = *groupCounter
			filtered = append(filtered, k)
			*groupCounter++

			// 如果不是最後一筆，找後續第一筆非目標擊殺
			if i < len(localKills)-1 {
				for j := i + 1; j < len(localKills); j++ {
					if localKills[j].IsTarget == 0 {
						nk := localKills[j]
						nk.GroupID = *groupCounter
						filtered = append(filtered, nk)
						*groupCounter++
						break
					}
				}
			}
		}
	}

	// 3. 合併到 allKills 並寫出 JSON (與 MkdirAll 相同路徑)
	*allKills = append(*allKills, filtered...)
	if data, err := json.MarshalIndent(*allKills, "", "  "); err == nil {
		_ = os.WriteFile("../output/kills/kills.json", data, 0644)
	}

	// 修正：加上 \n 才能正確換行
	fmt.Printf("✅ 完成 %s，擷取到目標+對應非目標共：%d 筆\n", filepath.Base(demPath), len(filtered))
	return len(filtered)
}

func main() {
	// 支援 flag 及環境變數
	targetPtr := flag.String("target", "", "comma-separated target players (fallback to TARGET_PLAYER env)")
	demPtr := flag.String("folder", "", "path to DEM folder (fallback to DEM_FOLDER env)")
	flag.Parse()

	targetEnv := *targetPtr
	if targetEnv == "" {
		targetEnv = os.Getenv("TARGET_PLAYER")
	}
	demFolder := *demPtr
	if demFolder == "" {
		demFolder = os.Getenv("DEM_FOLDER")
	}
	if targetEnv == "" || demFolder == "" {
		log.Fatal("❌ 請設定 TARGET_PLAYER 與 DEM_FOLDER 環境變數，或使用 -target 與 -folder 參數")
	}

	targets := map[string]bool{}
	for _, name := range strings.Split(targetEnv, ",") {
		targets[strings.ToLower(strings.TrimSpace(name))] = true
	}

	// 建立目錄
	os.MkdirAll("../output/kills", os.ModePerm)
	os.MkdirAll("../logs", os.ModePerm)

	killPath := "../output/kills/kills.json"
	logPath := "../logs/killlog.txt"

	// 載入歷史資料
	processedSet := loadProcessedDEMSet(logPath)
	var allKills []KillInfo
	groupCounter := 0
	if data, err := os.ReadFile(killPath); err == nil {
		_ = json.Unmarshal(data, &allKills)
		groupCounter = len(allKills)
	}

	// 處理 DEM: 使用 flag 或環境變數指定的 demFolder
	demFiles, _ := filepath.Glob(filepath.Join(demFolder, "*.dem"))
	for _, path := range demFiles {
		filename := filepath.Base(path)
		if processedSet[filename] {
			continue
		}
		info, _ := os.Stat(path)
		demoTime := info.ModTime().Format("2006-01-02")
		if count := processDemo(path, filename, demoTime, targets, &allKills, &groupCounter); count > 0 {
			appendToLog(logPath, filename)
		}
	}

	fmt.Println("🎉 所有未處理 DEM 已完成")
}
