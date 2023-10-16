package qlearn

import (
	"strconv"
	"strings"
)

type Agent struct {
	Nact     int
	InitValQ float64
	Epsilon  float64
	Alpha    float64
	Gamma    float64
	Q        map[int][]float64
	LenQ     int
}

func NewAgent() *Agent {
	return &Agent{
		Nact:     63,    // 投与の最大値
		InitValQ: -1e10, // なるべく負の値を小さくするのが目的
		Epsilon:  0.1,
		Alpha:    0.1,
		Gamma:    0.9,
		Q:        map[int][]float64{},
		LenQ:     0,
	}
}

func (e *Agent) Learn(s int, act int, rwd float64, next_s int) {
	e.checkAndAddObservation(s)
	e.checkAndAddObservation(next_s)

	target := float64(0)
	target = rwd + e.Gamma*maxValue(e.Q[next_s])

	e.Q[s][act] = (1-e.Alpha)*e.Q[s][act] + e.Alpha*target
}

func (e *Agent) GetQ(s int) []float64 {

	Q := []float64{}

	if _, isExist := e.Q[s]; isExist {
		Q = e.Q[s]
	} else {
		Q = nil
	}

	return Q
}

func (e *Agent) toStr(obs []int) string {
	strs := make([]string, len(obs))

	for i, o := range obs {
		strs[i] = strconv.Itoa(o)
	}
	return strings.Join(strs, ",")
}

func (e *Agent) checkAndAddObservation(s int) {
	if _, isExist := e.Q[s]; !isExist {
		e.Q[s] = make([]float64, e.Nact)
		for i := 0; i < e.Nact; i++ {
			e.Q[s][i] = e.InitValQ
		}
		e.LenQ++
	}
}

func maxIdx(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, v := range slice {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex
}

func maxValue(slice []float64) float64 {
	maxValue := slice[0]
	for _, v := range slice {
		if v > maxValue {
			maxValue = v
		}
	}
	return maxValue
}
