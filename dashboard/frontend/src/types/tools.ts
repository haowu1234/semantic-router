/**
 * Tool types for vLLM Semantic Router Dashboard
 */

export type ToolSource = 'builtin' | 'mcp';

export interface ToolParameter {
  name: string;
  type: 'string' | 'integer' | 'number' | 'boolean' | 'array' | 'object';
  description: string;
  required: boolean;
  default?: any;
  enum?: any[];
}

export interface Tool {
  id: string;
  name: string;
  description: string;
  source: ToolSource;
  mcp_server?: string;
  parameters: ToolParameter[];
  enabled: boolean;
  metadata?: Record<string, any>;
}

export interface ToolExecutionRequest {
  tool_id: string;
  arguments: Record<string, any>;
}

export interface ToolExecutionResult {
  success: boolean;
  result?: any;
  error?: string;
  duration_ms: number;
}

// MCP Types
export type MCPTransportType = 'stdio' | 'sse' | 'http';
export type MCPServerStatus = 'connected' | 'disconnected' | 'error' | 'connecting';

export interface MCPServer {
  id: string;
  name: string;
  transport_type: MCPTransportType;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  status: MCPServerStatus;
  error?: string;
  tool_count: number;
  created_at: string;
  updated_at: string;
}

export interface MCPToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
}

// Playground Tool Settings
export interface PlaygroundToolSettings {
  enabled: boolean;
  selectionMode: 'auto' | 'manual';
  selectedTools: string[];
  topK: number;
  similarityThreshold: number;
}

// Tool Call in Chat (OpenAI format)
export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolCallResult {
  tool_call_id: string;
  role: 'tool';
  content: string;
  duration_ms?: number;
  success?: boolean;
}

// Convert Tool to OpenAI format
export function toolToOpenAIFormat(tool: Tool): Record<string, any> {
  const properties: Record<string, any> = {};
  const required: string[] = [];

  for (const param of tool.parameters) {
    properties[param.name] = {
      type: param.type,
      description: param.description,
    };
    if (param.enum && param.enum.length > 0) {
      properties[param.name].enum = param.enum;
    }
    if (param.default !== undefined) {
      properties[param.name].default = param.default;
    }
    if (param.required) {
      required.push(param.name);
    }
  }

  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: 'object',
        properties,
        required,
      },
    },
  };
}
