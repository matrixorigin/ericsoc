// Package main runs the Chicago TNP two-stage recorded-tip ONNX model.
// Package main 运行 Chicago TNP 两阶段记录小费 ONNX 模型。
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// PresenceModel defines the first-stage classifier contract.
// PresenceModel 定义第一阶段分类器约定。
type PresenceModel struct {
	File                  string `json:"file"`
	InputName             string `json:"input_name"`
	ProbabilityOutputName string `json:"probability_output_name"`
	PositiveClassIndex    int    `json:"positive_class_index"`
}

// AmountModel defines the second-stage amount regressor contract.
// AmountModel 定义第二阶段金额回归器约定。
type AmountModel struct {
	File       string `json:"file"`
	InputName  string `json:"input_name"`
	OutputName string `json:"output_name"`
}

// Postprocessing defines the final expected-tip calculation.
// Postprocessing 定义最终期望小费计算。
type Postprocessing struct {
	TipCap float32 `json:"tip_cap"`
}

// Schema defines the ordered feature and ONNX model contracts.
// Schema 定义有序特征与 ONNX 模型约定。
type Schema struct {
	FeatureOrder   []string       `json:"feature_order"`
	PresenceModel  PresenceModel  `json:"presence_model"`
	AmountModel    AmountModel    `json:"amount_model"`
	Postprocessing Postprocessing `json:"postprocessing"`
}

// TestCase contains one prepared trip vector and optional reference outputs.
// TestCase 包含一条已准备的行程向量和可选参考输出。
type TestCase struct {
	CaseID                      string    `json:"case_id"`
	TripStartTimestamp          string    `json:"trip_start_timestamp"`
	PickupH3                    string    `json:"pickup_h3"`
	DropoffH3                   string    `json:"dropoff_h3"`
	Features                    []float32 `json:"features"`
	ActualRecordedTip           float32   `json:"actual_recorded_tip"`
	ONNXPythonTipProbability    float32   `json:"onnx_python_tip_probability"`
	ONNXPythonPositiveTipAmount float32   `json:"onnx_python_positive_tip_amount"`
	ONNXPythonExpectedTip       float32   `json:"onnx_python_expected_tip"`
}

// Prediction contains the three outputs of the two-stage model.
// Prediction 保存两阶段模型的三个输出。
type Prediction struct {
	TipProbability    float32
	PositiveTipAmount float32
	ExpectedTip       float32
}

// readJSON reads one UTF-8 JSON file into a typed destination.
// readJSON 读取 UTF-8 JSON 文件并解析到指定结构。
func readJSON(path string, destination any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, destination)
}

// flattenFeatures validates and flattens the prepared feature vectors.
// flattenFeatures 验证并展开已准备的特征向量。
func flattenFeatures(cases []TestCase, featureCount int) ([]float32, error) {
	if len(cases) == 0 {
		return nil, errors.New("test case file is empty / 测试样本文件为空")
	}
	if featureCount == 0 {
		return nil, errors.New("feature schema is empty / 特征结构为空")
	}

	flat := make([]float32, 0, len(cases)*featureCount)
	for _, testCase := range cases {
		if len(testCase.Features) != featureCount {
			return nil, fmt.Errorf(
				"%s has %d features; schema requires %d / 特征数量与结构不一致",
				testCase.CaseID,
				len(testCase.Features),
				featureCount,
			)
		}
		for featureIndex, value := range testCase.Features {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf(
					"%s feature %d is not finite / 特征不是有限数值",
					testCase.CaseID,
					featureIndex,
				)
			}
		}
		flat = append(flat, testCase.Features...)
	}
	return flat, nil
}

// runPresenceModel returns the positive recorded-tip probability.
// runPresenceModel 返回记录正小费的概率。
func runPresenceModel(modelPath string, schema Schema, cases []TestCase, flat []float32) ([]float32, error) {
	featureCount := len(schema.FeatureOrder)
	inputTensor, err := ort.NewTensor(
		ort.NewShape(int64(len(cases)), int64(featureCount)),
		flat,
	)
	if err != nil {
		return nil, fmt.Errorf("create classifier input / 创建分类器输入: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(int64(len(cases)), 2),
	)
	if err != nil {
		return nil, fmt.Errorf("create classifier output / 创建分类器输出: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{schema.PresenceModel.InputName},
		[]string{schema.PresenceModel.ProbabilityOutputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create classifier session / 创建分类器会话: %w", err)
	}
	defer session.Destroy()
	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run classifier / 运行分类器: %w", err)
	}

	matrix := outputTensor.GetData()
	probability := make([]float32, len(cases))
	for index := range cases {
		value := matrix[index*2+schema.PresenceModel.PositiveClassIndex]
		probability[index] = min(max(value, 0), 1)
	}
	return probability, nil
}

// runAmountModel returns the log-scale positive-tip amount output.
// runAmountModel 返回对数尺度的正小费金额输出。
func runAmountModel(modelPath string, schema Schema, cases []TestCase, flat []float32) ([]float32, error) {
	featureCount := len(schema.FeatureOrder)
	inputTensor, err := ort.NewTensor(
		ort.NewShape(int64(len(cases)), int64(featureCount)),
		flat,
	)
	if err != nil {
		return nil, fmt.Errorf("create amount input / 创建金额模型输入: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(int64(len(cases)), 1),
	)
	if err != nil {
		return nil, fmt.Errorf("create amount output / 创建金额模型输出: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{schema.AmountModel.InputName},
		[]string{schema.AmountModel.OutputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create amount session / 创建金额模型会话: %w", err)
	}
	defer session.Destroy()
	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run amount model / 运行金额模型: %w", err)
	}
	return append([]float32(nil), outputTensor.GetData()...), nil
}

// runModels executes both stages and calculates expected recorded tip.
// runModels 执行两个阶段并计算记录小费期望值。
func runModels(modelDir string, runtimePath string, schema Schema, cases []TestCase) ([]Prediction, error) {
	flat, err := flattenFeatures(cases, len(schema.FeatureOrder))
	if err != nil {
		return nil, err
	}

	ort.SetSharedLibraryPath(runtimePath)
	if err := ort.InitializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime / 初始化 ONNX Runtime: %w", err)
	}
	defer ort.DestroyEnvironment()

	probability, err := runPresenceModel(
		modelDir+"/"+schema.PresenceModel.File, schema, cases, flat,
	)
	if err != nil {
		return nil, err
	}
	logAmount, err := runAmountModel(
		modelDir+"/"+schema.AmountModel.File, schema, cases, flat,
	)
	if err != nil {
		return nil, err
	}

	predictions := make([]Prediction, len(cases))
	for index := range cases {
		amount := float32(math.Expm1(float64(logAmount[index])))
		amount = min(max(amount, 0), schema.Postprocessing.TipCap)
		predictions[index] = Prediction{
			TipProbability:    probability[index],
			PositiveTipAmount: amount,
			ExpectedTip:       probability[index] * amount,
		}
	}
	return predictions, nil
}

// main parses flags, runs inference, and optionally verifies Python parity.
// main 解析参数、执行推理，并按需验证 Python 一致性。
func main() {
	defaultRuntime := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY")
	if defaultRuntime == "" {
		defaultRuntime = "runtime/macos-arm64/libonnxruntime.dylib"
	}

	modelDir := flag.String("model-dir", "model", "ONNX model directory / ONNX 模型目录")
	schemaPath := flag.String("schema", "model/feature_schema.json", "feature schema path / 特征结构路径")
	casesPath := flag.String("cases", "testdata/parity_test_cases.json", "input case path / 输入样本路径")
	runtimePath := flag.String("onnxruntime", defaultRuntime, "ONNX Runtime shared library / ONNX Runtime 动态库")
	tolerance := flag.Float64("tolerance", 0.001, "maximum parity difference / 最大一致性误差")
	verify := flag.Bool("verify", true, "compare with stored Python references / 与 Python 参考结果对比")
	flag.Parse()

	var schema Schema
	if err := readJSON(*schemaPath, &schema); err != nil {
		panic(fmt.Errorf("read schema / 读取特征结构: %w", err))
	}
	var cases []TestCase
	if err := readJSON(*casesPath, &cases); err != nil {
		panic(fmt.Errorf("read cases / 读取输入样本: %w", err))
	}

	predictions, err := runModels(*modelDir, *runtimePath, schema, cases)
	if err != nil {
		panic(err)
	}

	maxDifference := 0.0
	for index, prediction := range predictions {
		probabilityDifference := math.Abs(float64(
			prediction.TipProbability - cases[index].ONNXPythonTipProbability,
		))
		amountDifference := math.Abs(float64(
			prediction.PositiveTipAmount - cases[index].ONNXPythonPositiveTipAmount,
		))
		expectedDifference := math.Abs(float64(
			prediction.ExpectedTip - cases[index].ONNXPythonExpectedTip,
		))
		maxDifference = max(
			maxDifference,
			probabilityDifference,
			amountDifference,
			expectedDifference,
		)

		if *verify {
			fmt.Printf(
				"%s | p_tip=%.6f | positive_amount=%.6f | expected_tip=%.6f | max_diff=%.8f\n",
				cases[index].CaseID,
				prediction.TipProbability,
				prediction.PositiveTipAmount,
				prediction.ExpectedTip,
				max(probabilityDifference, amountDifference, expectedDifference),
			)
		} else {
			fmt.Printf(
				"%s | pickup=%s | dropoff=%s | p_tip=%.4f | positive_amount=$%.2f | expected_tip=$%.2f\n",
				cases[index].CaseID,
				cases[index].PickupH3,
				cases[index].DropoffH3,
				prediction.TipProbability,
				prediction.PositiveTipAmount,
				prediction.ExpectedTip,
			)
		}
	}

	if !*verify {
		return
	}
	fmt.Printf("max absolute difference / 最大绝对差异: %.10f\n", maxDifference)
	if maxDifference > *tolerance {
		fmt.Fprintf(os.Stderr, "FAIL: difference exceeds tolerance %.6f / 差异超过容限\n", *tolerance)
		os.Exit(1)
	}
	fmt.Println("PASS: Go and Python ONNX Runtime predictions match within tolerance.")
	fmt.Println("通过：Go 与 Python ONNX Runtime 预测在容限内一致。")
}
