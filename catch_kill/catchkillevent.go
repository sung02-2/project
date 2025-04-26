package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
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
	DemoID     int    `json:"demo_id"`
}

type Label struct {
	GroupID int `json:"group_id"`
	Player  int `json:"player?"`
}

const maxGroup = 32

var (
	killsBuffer  []KillInfo
	labelsBuffer []Label
	outputIndex  = 1
	groupCounter = 0
)

func loadPartialData() {
	kf, err := ioutil.ReadFile("logs/partial_kills.json")
	if err == nil {
		json.Unmarshal(kf, &killsBuffer)
	}
	lf, err := ioutil.ReadFile("logs/partial_labels.json")
	if err == nil {
		json.Unmarshal(lf, &labelsBuffer)
	}
	// 讀取最後使用過的 index
	idxBytes, err := ioutil.ReadFile("logs/last_output_index.txt")
	if err == nil {
		if idx, err := strconv.Atoi(string(idxBytes)); err == nil && idx > 0 {
			outputIndex = idx + 1
		}
	}
}

func savePartialData() {
	kb, _ := json.MarshalIndent(killsBuffer, "", "  ")
	_ = ioutil.WriteFile("logs/partial_kills.json", kb, 0644)
	lb, _ := json.MarshalIndent(labelsBuffer, "", "  ")
	_ = ioutil.WriteFile("logs/partial_labels.json", lb, 0644)
	_ = ioutil.WriteFile("logs/last_output_index.txt", []byte(strconv.Itoa(outputIndex-1)), 0644)
}

func loadProcessedDems() map[string]bool {
	data, err := ioutil.ReadFile("logs/processed_dems.json")
	if err != nil {
		return map[string]bool{}
	}
	var list []string
	_ = json.Unmarshal(data, &list)
	m := map[string]bool{}
	for _, name := range list {
		m[name] = true
	}
	return m
}

func saveProcessedDems(processed map[string]bool) {
	var list []string
	for name := range processed {
		list = append(list, name)
	}
	data, _ := json.MarshalIndent(list, "", "  ")
	_ = ioutil.WriteFile("logs/processed_dems.json", data, 0644)
}

func flushIfFull() {
	for len(killsBuffer) >= maxGroup && len(labelsBuffer) >= maxGroup {
		batchKills := killsBuffer[:maxGroup]
		batchLabels := labelsBuffer[:maxGroup]

		kPath := fmt.Sprintf("output/kills/kills%d.json", outputIndex)
		lPath := fmt.Sprintf("output/labels/labels%d.csv", outputIndex)

		kb, _ := json.MarshalIndent(batchKills, "", "  ")
		_ = ioutil.WriteFile(kPath, kb, 0644)

		f, _ := os.Create(lPath)
		w := csv.NewWriter(f)
		w.Write([]string{"GroupID", "player?", "DemoID"}) // ← 新增 DemoID 欄位

		for i, lbl := range batchLabels {
			demoID := batchKills[i].DemoID
			w.Write([]string{
				strconv.Itoa(lbl.GroupID),
				strconv.Itoa(lbl.Player),
				strconv.Itoa(demoID),
			})
		}
		w.Flush()
		f.Close()

		fmt.Printf("✅ 輸出 kills%d.json / labels%d.csv\n", outputIndex, outputIndex)
		outputIndex++
		killsBuffer = killsBuffer[maxGroup:]
		labelsBuffer = labelsBuffer[maxGroup:]
	}
}

func main() {
	targetEnv := os.Getenv("TARGET_PLAYER")
	demFolder := os.Getenv("DEM_FOLDER")
	if targetEnv == "" || demFolder == "" {
		log.Fatal("請設定 TARGET_PLAYER 與 DEM_FOLDER 環境變數")
	}

	targets := map[string]bool{}
	for _, name := range strings.Split(targetEnv, ",") {
		targets[strings.ToLower(strings.TrimSpace(name))] = true
	}

	os.MkdirAll("output/kills", os.ModePerm)
	os.MkdirAll("output/labels", os.ModePerm)
	os.MkdirAll("logs", os.ModePerm)

	loadPartialData()
	processedDems := loadProcessedDems()

	demFiles, _ := filepath.Glob(filepath.Join(demFolder, "game*.dem"))
	for _, path := range demFiles {
		filename := filepath.Base(path)
		if processedDems[filename] {
			fmt.Printf("⚠️ %s 已處理過，略過。\n", filename)
			continue
		}

		demoID := 0
		fmt.Sscanf(filename, "game%d.dem", &demoID)

		f, err := os.Open(path)
		if err != nil {
			log.Println("無法開啟 DEM:", path)
			continue
		}
		defer f.Close()

		p := dem.NewParser(f)
		defer p.Close()

		p.RegisterEventHandler(func(e events.Kill) {
			if e.Killer == nil {
				return
			}
			weapon := e.Killer.ActiveWeapon()
			if weapon == nil {
				return
			}
			killerName := e.Killer.Name
			isTarget := targets[strings.ToLower(killerName)]

			if !isTarget {
				count := 0
				for _, lbl := range labelsBuffer {
					if lbl.Player == 0 {
						count++
					}
				}
				if count >= len(labelsBuffer)/2 {
					return
				}
			}

			info := KillInfo{
				GroupID:    groupCounter,
				Tick:       p.CurrentFrame(),
				KillerName: killerName,
				VictimName: e.Victim.Name,
				WeaponName: weapon.Type.String(),
				DemoID:     demoID,
			}
			label := Label{
				GroupID: groupCounter,
				Player:  0,
			}
			if isTarget {
				label.Player = 1
			}

			killsBuffer = append(killsBuffer, info)
			labelsBuffer = append(labelsBuffer, label)
			groupCounter++
			flushIfFull()
		})

		p.ParseToEnd()
		processedDems[filename] = true
	}

	saveProcessedDems(processedDems)
	savePartialData()
	fmt.Printf("📌 剩餘未滿 %d 筆資料，已寫入 partial 檔案，下次執行將補齊\n", len(killsBuffer))
}
