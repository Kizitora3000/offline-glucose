package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"rl/qlearn"
	"strconv"
)

func main() {
	Agt := qlearn.NewAgent()

	dirname := "./proprocessed_diabetes_dataset"

	files, err := os.ReadDir(dirname)
	if err != nil {
		panic(err)
	}

	mx := 0.0
	for _, file := range files {
		filename := filepath.Join(dirname, file.Name())
		file, err := os.Open(filename)

		// open csv
		if err != nil {
			fmt.Printf("Error opening file %s: %v\n", filename, err)
			return
		}
		defer file.Close()

		r := csv.NewReader(file)
		records, err := r.ReadAll()
		if err != nil {
			fmt.Printf("Error reading CSV %s: %v\n", filename, err)
			return
		}

		// Exclude the last row
		records = records[:len(records)-1]

		for _, record := range records {
			status, _ := strconv.Atoi(record[1])
			action, _ := strconv.Atoi(record[2])
			rwd, _ := strconv.ParseFloat(record[3], 64)
			next_status, _ := strconv.Atoi(record[4])
			mx = math.Max(mx, float64(action))

			Agt.Learn(status, action, rwd, next_status)
		}
	}
	for key, values := range Agt.Q {
		fmt.Printf("Key: %s, Values: %v\n", key, values)
	}
}
