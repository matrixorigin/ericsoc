// Package main runs the Chicago TNP H3 next-hour ONNX model.
// Package main 运行 Chicago TNP H3 下一小时 ONNX 模型。
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

// Schema defines the ordered ONNX input contract.
// Schema 定义有序的 ONNX 输入约定。
type Schema struct {
	InputName    string   `json:"input_name"`
	OutputName   string   `json:"output_name"`
	FeatureOrder []string `json:"feature_order"`
}

// TestCase contains one prepared feature vector and optional reference output.
// TestCase 包含一条已准备的特征向量和可选参考结果。
type TestCase struct {
	CaseID               string    `json:"case_id"`
	H3                   string    `json:"h3"`
	FeatureTimestamp     string    `json:"feature_timestamp"`
	TargetTimestamp      string    `json:"target_timestamp"`
	Features             []float32 `json:"features"`
	ActualPickups        float32   `json:"actual_pickups"`
	PythonPrediction     float32   `json:"python_prediction"`
	ONNXPythonPrediction float32   `json:"onnx_python_prediction"`
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

// runModel validates inputs and performs one batched ONNX inference call.
// runModel 验证输入并执行一次批量 ONNX 推理。
func runModel(modelPath string, runtimePath string, schema Schema, cases []TestCase) ([]float32, error) {
	if len(cases) == 0 {
		return nil, errors.New("test case file is empty / 测试样本文件为空")
	}
	featureCount := len(schema.FeatureOrder)
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

	ort.SetSharedLibraryPath(runtimePath)
	if err := ort.InitializeEnvironment(ort.WithLogLevelWarning()); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime / 初始化 ONNX Runtime: %w", err)
	}
	defer ort.DestroyEnvironment()

	inputTensor, err := ort.NewTensor(
		ort.NewShape(int64(len(cases)), int64(featureCount)),
		flat,
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor / 创建输入张量: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(len(cases)), 1))
	if err != nil {
		return nil, fmt.Errorf("create output tensor / 创建输出张量: %w", err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{schema.InputName},
		[]string{schema.OutputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create ONNX session / 创建 ONNX 会话: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run ONNX model / 运行 ONNX 模型: %w", err)
	}

	result := append([]float32(nil), outputTensor.GetData()...)
	for index := range result {
		if result[index] < 0 {
			result[index] = 0
		}
	}
	return result, nil
}

// main parses command-line flags, runs inference, and optionally verifies parity.
// main 解析命令行参数、执行推理，并按需验证跨语言一致性。
func main() {
	defaultRuntime := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY")
	if defaultRuntime == "" {
		defaultRuntime = "runtime/macos-arm64/libonnxruntime.dylib"
	}

	modelPath := flag.String("model", "model/h3_next_hour_pickups.onnx", "ONNX model path / ONNX 模型路径")
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

	predictions, err := runModel(*modelPath, *runtimePath, schema, cases)
	if err != nil {
		panic(err)
	}

	maxDifference := 0.0
	for index, prediction := range predictions {
		difference := math.Abs(float64(prediction - cases[index].ONNXPythonPrediction))
		if difference > maxDifference {
			maxDifference = difference
		}
		if *verify {
			fmt.Printf(
				"%s | H3=%s | next_hour=%.6f | python_onnx=%.6f | diff=%.8f\n",
				cases[index].CaseID,
				cases[index].H3,
				prediction,
				cases[index].ONNXPythonPrediction,
				difference,
			)
		} else {
			fmt.Printf(
				"%s | H3=%s | feature_time=%s | predicted_next_hour_pickups=%.6f\n",
				cases[index].CaseID,
				cases[index].H3,
				cases[index].FeatureTimestamp,
				prediction,
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
