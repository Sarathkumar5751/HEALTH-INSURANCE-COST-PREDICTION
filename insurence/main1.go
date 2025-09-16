package main

import (
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// ---------------------- Data Structures ----------------------

type DataSet struct {
	X        [][]float64
	Y        []float64
	ColNames []string
}

type TreeNode struct {
	Feature   int
	Threshold float64
	Left      *TreeNode
	Right     *TreeNode
	Value     float64
}

type RandomForest struct {
	Trees           []*TreeNode
	NFeatures       int
	MaxDepth        int
	MinSamplesSplit int
	NEstimators     int
	Seed            int64
}

// ---------------------- Data Handling ----------------------

func ReadCSV(path string) (*DataSet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("CSV must contain header and at least one row")
	}

	header := records[0]
	ncols := len(header)
	nrows := len(records) - 1
	X := make([][]float64, nrows)
	Y := make([]float64, nrows)

	for i := 0; i < nrows; i++ {
		X[i] = make([]float64, ncols-1)
		row := records[i+1]
		for j := 0; j < ncols; j++ {
			val := strings.TrimSpace(row[j])
			if j == ncols-1 {
				fv, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return nil, fmt.Errorf("failed to parse target on row %d: %v", i+2, err)
				}
				Y[i] = fv
			} else {
				if fv, err := strconv.ParseFloat(val, 64); err == nil {
					X[i][j] = fv
				} else {
					l := strings.ToLower(val)
					switch l {
					case "male":
						X[i][j] = 0.0
					case "female":
						X[i][j] = 1.0
					case "yes":
						X[i][j] = 1.0
					case "no":
						X[i][j] = 0.0
					case "northwest", "nw":
						X[i][j] = 0.0
					case "northeast", "ne":
						X[i][j] = 1.0
					case "southeast", "se":
						X[i][j] = 2.0
					case "southwest", "sw":
						X[i][j] = 3.0
					default:
						X[i][j] = float64(len(l))
					}
				}
			}
		}
	}
	ds := &DataSet{X: X, Y: Y, ColNames: header[:ncols-1]}
	return ds, nil
}

func TrainTestSplit(ds *DataSet, ratio float64, seed int64) (train *DataSet, test *DataSet) {
	n := len(ds.Y)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	rnd := rand.New(rand.NewSource(seed))
	rnd.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

	nTrain := int(math.Floor(ratio * float64(n)))
	Xtr := make([][]float64, nTrain)
	Ytr := make([]float64, nTrain)
	Xte := make([][]float64, n-nTrain)
	Yte := make([]float64, n-nTrain)

	for i := 0; i < nTrain; i++ {
		Xtr[i] = ds.X[idx[i]]
		Ytr[i] = ds.Y[idx[i]]
	}
	for i := nTrain; i < n; i++ {
		Xte[i-nTrain] = ds.X[idx[i]]
		Yte[i-nTrain] = ds.Y[idx[i]]
	}
	train = &DataSet{X: Xtr, Y: Ytr, ColNames: ds.ColNames}
	test = &DataSet{X: Xte, Y: Yte, ColNames: ds.ColNames}
	return
}

// ---------------------- Tree Building ----------------------

func BuildTree(X [][]float64, Y []float64, maxDepth int, minSamples int, nFeatures int, rnd *rand.Rand) *TreeNode {
	if len(Y) <= minSamples || maxDepth == 0 {
		return &TreeNode{Feature: -1, Value: mean(Y)}
	}
	nSamples := len(Y)
	nCols := len(X[0])

	featIdx := randPermSubset(nCols, nFeatures, rnd)
	bestFeat := -1
	bestThresh := 0.0
	bestScore := math.Inf(1)
	var leftIdx, rightIdx []int

	for _, f := range featIdx {
		vals := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			vals[i] = X[i][f]
		}
		unique := uniqueFloats(vals)
		if len(unique) == 1 {
			continue
		}
		sortFloatSlice(unique)
		for i := 0; i < len(unique)-1; i++ {
			thr := (unique[i] + unique[i+1]) / 2.0
			lIdx := make([]int, 0, nSamples/2)
			rIdx := make([]int, 0, nSamples/2)
			for j := 0; j < nSamples; j++ {
				if X[j][f] <= thr {
					lIdx = append(lIdx, j)
				} else {
					rIdx = append(rIdx, j)
				}
			}
			if len(lIdx) == 0 || len(rIdx) == 0 {
				continue
			}
			lY := indexSlice(Y, lIdx)
			rY := indexSlice(Y, rIdx)
			score := float64(len(lY))*variance(lY) + float64(len(rY))*variance(rY)
			if score < bestScore {
				bestScore = score
				bestFeat = f
				bestThresh = thr
				leftIdx = lIdx
				rightIdx = rIdx
			}
		}
	}

	if bestFeat == -1 {
		return &TreeNode{Feature: -1, Value: mean(Y)}
	}
	Xl, Yl := subsetXY(X, Y, leftIdx)
	Xr, Yr := subsetXY(X, Y, rightIdx)
	left := BuildTree(Xl, Yl, maxDepth-1, minSamples, nFeatures, rnd)
	right := BuildTree(Xr, Yr, maxDepth-1, minSamples, nFeatures, rnd)
	return &TreeNode{Feature: bestFeat, Threshold: bestThresh, Left: left, Right: right}
}

// ---------------------- RandomForest ----------------------

func (rf *RandomForest) fitSingleTree(X [][]float64, Y []float64, rnd *rand.Rand) *TreeNode {
	n := len(Y)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = rnd.Intn(n)
	}
	Xs, Ys := subsetXY(X, Y, idx)
	return BuildTree(Xs, Ys, rf.MaxDepth, rf.MinSamplesSplit, rf.NFeatures, rnd)
}

func (rf *RandomForest) Fit(X [][]float64, Y []float64) {
	rf.Trees = make([]*TreeNode, rf.NEstimators)
	rnd := rand.New(rand.NewSource(rf.Seed))
	for i := 0; i < rf.NEstimators; i++ {
		rf.Trees[i] = rf.fitSingleTree(X, Y, rnd)
	}
}

func (rf *RandomForest) PredictRow(x []float64) float64 {
	sum := 0.0
	for _, t := range rf.Trees {
		sum += predictTree(t, x)
	}
	return sum / float64(len(rf.Trees))
}

func (rf *RandomForest) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := 0; i < len(X); i++ {
		out[i] = rf.PredictRow(X[i])
	}
	return out
}

// ---------------------- Save/Load ----------------------

func (rf *RandomForest) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(rf)
}

func LoadModel(path string) (*RandomForest, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var rf RandomForest
	if err := dec.Decode(&rf); err != nil {
		return nil, err
	}
	return &rf, nil
}

// ---------------------- Utilities ----------------------

func predictTree(node *TreeNode, x []float64) float64 {
	if node.Feature == -1 || node.Left == nil || node.Right == nil {
		return node.Value
	}
	if x[node.Feature] <= node.Threshold {
		return predictTree(node.Left, x)
	}
	return predictTree(node.Right, x)
}

func mean(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range arr {
		sum += v
	}
	return sum / float64(len(arr))
}

func variance(arr []float64) float64 {
	if len(arr) <= 1 {
		return 0
	}
	m := mean(arr)
	s := 0.0
	for _, v := range arr {
		d := v - m
		s += d * d
	}
	return s / float64(len(arr))
}

func indexSlice(arr []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, v := range idx {
		out[i] = arr[v]
	}
	return out
}

func subsetXY(X [][]float64, Y []float64, idx []int) ([][]float64, []float64) {
	Xs := make([][]float64, len(idx))
	Ys := make([]float64, len(idx))
	for i, j := range idx {
		Xs[i] = X[j]
		Ys[i] = Y[j]
	}
	return Xs, Ys
}

func uniqueFloats(a []float64) []float64 {
	m := make(map[float64]struct{})
	for _, v := range a {
		m[v] = struct{}{}
	}
	out := make([]float64, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func sortFloatSlice(a []float64) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}

func randPermSubset(n int, k int, rnd *rand.Rand) []int {
	if k >= n {
		out := make([]int, n)
		for i := 0; i < n; i++ {
			out[i] = i
		}
		return out
	}
	perm := rnd.Perm(n)
	return perm[:k]
}

func MSE(yTrue, yPred []float64) float64 {
	n := len(yTrue)
	if n == 0 {
		return 0
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		d := yTrue[i] - yPred[i]
		sum += d * d
	}
	return sum / float64(n)
}

// ---------------------- Main ----------------------

func main() {
	csvPath := flag.String("csv", "insurance.csv", "Path to insurance.csv (header required)")
	modelOut := flag.String("model", "rf_model.gob", "Output model file")
	flag.Parse()

	fmt.Println("Reading CSV:", *csvPath)
	ds, err := ReadCSV(*csvPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d rows, %d features\n", len(ds.Y), len(ds.X[0]))

	train, test := TrainTestSplit(ds, 0.8, time.Now().UnixNano())
	fmt.Printf("Train: %d rows, Test: %d rows\n", len(train.Y), len(test.Y))

	rf := &RandomForest{
		NFeatures:       3,
		MaxDepth:        6,
		MinSamplesSplit: 5,
		NEstimators:     40,
		Seed:            time.Now().UnixNano(),
	}

	fmt.Println("Training Random Forest...")
	rf.Fit(train.X, train.Y)

	fmt.Println("Saving model to", *modelOut)
	if err := rf.Save(*modelOut); err != nil {
		log.Println("Warning: failed to save model:", err)
	}

	fmt.Println("Evaluating on test set...")
	preds := rf.Predict(test.X)
	mse := MSE(test.Y, preds)
	rmse := math.Sqrt(mse)
	fmt.Printf("Test RMSE: %.4f\n", rmse)
	fmt.Println("Done.")
}
