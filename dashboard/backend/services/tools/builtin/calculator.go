package builtin

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// Calculator is a built-in tool for mathematical calculations
type Calculator struct{}

// Name returns the tool name (matches tools_db.json)
func (c *Calculator) Name() string {
	return "calculate"
}

// Description returns the tool description
func (c *Calculator) Description() string {
	return "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), " +
		"power (**), modulo (%), parentheses, and common functions like sqrt, sin, cos, tan, log, abs, ceil, floor."
}

// Parameters returns the tool parameters
func (c *Calculator) Parameters() []models.ToolParameter {
	return []models.ToolParameter{
		{
			Name:        "expression",
			Type:        "string",
			Description: "The mathematical expression to evaluate, e.g., 'sqrt(16) + 2 * 3' or '(10 + 5) * 2'",
			Required:    true,
		},
	}
}

// Execute runs the calculator with the given expression
func (c *Calculator) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	expr, ok := args["expression"].(string)
	if !ok {
		return nil, fmt.Errorf("expression must be a string")
	}

	// Preprocess expression: replace function calls with values
	processedExpr := preprocessExpression(expr)

	result, err := evaluateExpression(processedExpr)
	if err != nil {
		return nil, fmt.Errorf("evaluation error: %v", err)
	}

	return map[string]interface{}{
		"expression": expr,
		"result":     result,
	}, nil
}

// preprocessExpression handles function calls like sqrt(16), sin(0), etc.
func preprocessExpression(expr string) string {
	// Define supported functions
	functions := map[string]func(float64) float64{
		"sqrt":  math.Sqrt,
		"sin":   math.Sin,
		"cos":   math.Cos,
		"tan":   math.Tan,
		"log":   math.Log,
		"log10": math.Log10,
		"abs":   math.Abs,
		"ceil":  math.Ceil,
		"floor": math.Floor,
		"exp":   math.Exp,
	}

	result := expr

	// Process each function
	for name, fn := range functions {
		for {
			idx := strings.Index(result, name+"(")
			if idx == -1 {
				break
			}

			// Find matching closing parenthesis
			start := idx + len(name) + 1
			depth := 1
			end := start
			for end < len(result) && depth > 0 {
				if result[end] == '(' {
					depth++
				} else if result[end] == ')' {
					depth--
				}
				end++
			}

			if depth != 0 {
				break
			}

			// Extract and evaluate the inner expression
			innerExpr := result[start : end-1]
			innerResult, err := evaluateExpression(preprocessExpression(innerExpr))
			if err != nil {
				break
			}

			// Apply the function
			funcResult := fn(innerResult)

			// Replace in result
			result = result[:idx] + fmt.Sprintf("%v", funcResult) + result[end:]
		}
	}

	// Handle pow(base, exp) separately
	for {
		idx := strings.Index(result, "pow(")
		if idx == -1 {
			break
		}

		start := idx + 4
		depth := 1
		end := start
		for end < len(result) && depth > 0 {
			if result[end] == '(' {
				depth++
			} else if result[end] == ')' {
				depth--
			}
			end++
		}

		if depth != 0 {
			break
		}

		innerExpr := result[start : end-1]
		parts := strings.SplitN(innerExpr, ",", 2)
		if len(parts) != 2 {
			break
		}

		base, err1 := evaluateExpression(preprocessExpression(strings.TrimSpace(parts[0])))
		exp, err2 := evaluateExpression(preprocessExpression(strings.TrimSpace(parts[1])))
		if err1 != nil || err2 != nil {
			break
		}

		powResult := math.Pow(base, exp)
		result = result[:idx] + fmt.Sprintf("%v", powResult) + result[end:]
	}

	// Replace ** with ^ for power (then handle in evaluator)
	result = strings.ReplaceAll(result, "**", "^")

	return result
}

// evaluateExpression evaluates a simple arithmetic expression
func evaluateExpression(expr string) (float64, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return 0, fmt.Errorf("empty expression")
	}

	// Try to parse as a simple number first
	if val, err := strconv.ParseFloat(expr, 64); err == nil {
		return val, nil
	}

	// Handle negative numbers at the start
	if strings.HasPrefix(expr, "-") {
		result, err := evaluateExpression(expr[1:])
		if err != nil {
			return 0, err
		}
		return -result, nil
	}

	// Use Go's parser for basic arithmetic
	node, err := parser.ParseExpr(expr)
	if err != nil {
		// Try replacing ^ with power operation manually
		if strings.Contains(expr, "^") {
			return evaluatePower(expr)
		}
		return 0, fmt.Errorf("parse error: %v", err)
	}

	return evalNode(node)
}

// evaluatePower handles expressions with ^ for power
func evaluatePower(expr string) (float64, error) {
	// Find the last ^ to handle right-to-left associativity
	idx := strings.LastIndex(expr, "^")
	if idx == -1 {
		return evaluateExpression(strings.ReplaceAll(expr, "^", ""))
	}

	base, err := evaluateExpression(expr[:idx])
	if err != nil {
		return 0, err
	}

	exp, err := evaluateExpression(expr[idx+1:])
	if err != nil {
		return 0, err
	}

	return math.Pow(base, exp), nil
}

// evalNode evaluates an AST node
func evalNode(node ast.Expr) (float64, error) {
	switch n := node.(type) {
	case *ast.BasicLit:
		return strconv.ParseFloat(n.Value, 64)

	case *ast.ParenExpr:
		return evalNode(n.X)

	case *ast.UnaryExpr:
		x, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case token.SUB:
			return -x, nil
		case token.ADD:
			return x, nil
		default:
			return 0, fmt.Errorf("unsupported unary operator: %v", n.Op)
		}

	case *ast.BinaryExpr:
		x, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		y, err := evalNode(n.Y)
		if err != nil {
			return 0, err
		}

		switch n.Op {
		case token.ADD:
			return x + y, nil
		case token.SUB:
			return x - y, nil
		case token.MUL:
			return x * y, nil
		case token.QUO:
			if y == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return x / y, nil
		case token.REM:
			return math.Mod(x, y), nil
		default:
			return 0, fmt.Errorf("unsupported binary operator: %v", n.Op)
		}

	default:
		return 0, fmt.Errorf("unsupported expression type: %T", node)
	}
}
