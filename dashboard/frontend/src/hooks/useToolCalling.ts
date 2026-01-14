/**
 * useToolCalling hook - Manages tool calling flow in chat
 */

import { useState, useCallback, useRef } from 'react';
import { Tool, ToolCall, ToolCallResult, toolToOpenAIFormat } from '../types/tools';

interface UseToolCallingOptions {
  maxIterations?: number;  // Max tool calling rounds to prevent infinite loops
  onToolCallStart?: (toolCall: ToolCall) => void;
  onToolCallComplete?: (result: ToolCallResult) => void;
  onError?: (error: string) => void;
}

interface ToolCallingState {
  isExecutingTools: boolean;
  currentToolCalls: ToolCall[];
  toolResults: ToolCallResult[];
  iterationCount: number;
}

export function useToolCalling(options: UseToolCallingOptions = {}) {
  const { maxIterations = 5, onToolCallStart, onToolCallComplete, onError } = options;
  
  const [state, setState] = useState<ToolCallingState>({
    isExecutingTools: false,
    currentToolCalls: [],
    toolResults: [],
    iterationCount: 0,
  });
  
  const enabledToolsRef = useRef<Tool[]>([]);

  // Fetch enabled tools from API
  const fetchEnabledTools = useCallback(async (): Promise<Tool[]> => {
    try {
      const response = await fetch('/api/tools');
      if (!response.ok) {
        throw new Error('Failed to fetch tools');
      }
      const data = await response.json();
      const tools = (data.tools || []).filter((t: Tool) => t.enabled);
      enabledToolsRef.current = tools;
      return tools;
    } catch (error) {
      console.error('Error fetching tools:', error);
      return [];
    }
  }, []);

  // Convert tools to OpenAI format for API request
  const getToolsForRequest = useCallback((): Record<string, unknown>[] => {
    return enabledToolsRef.current.map(tool => toolToOpenAIFormat(tool));
  }, []);

  // Execute a single tool
  const executeTool = useCallback(async (toolCall: ToolCall): Promise<ToolCallResult> => {
    onToolCallStart?.(toolCall);
    
    const startTime = Date.now();
    
    try {
      let args: Record<string, unknown>;
      try {
        args = JSON.parse(toolCall.function.arguments);
      } catch {
        throw new Error(`Invalid tool arguments: ${toolCall.function.arguments}`);
      }

      const response = await fetch('/api/tools/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_id: toolCall.function.name,
          arguments: args,
        }),
      });

      const data = await response.json();
      const duration_ms = Date.now() - startTime;

      const result: ToolCallResult = {
        tool_call_id: toolCall.id,
        role: 'tool',
        content: data.success ? JSON.stringify(data.result) : `Error: ${data.error}`,
        duration_ms,
        success: data.success,
      };

      onToolCallComplete?.(result);
      return result;
    } catch (error) {
      const duration_ms = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      onError?.(errorMessage);
      
      return {
        tool_call_id: toolCall.id,
        role: 'tool',
        content: `Error executing tool: ${errorMessage}`,
        duration_ms,
        success: false,
      };
    }
  }, [onToolCallStart, onToolCallComplete, onError]);

  // Execute multiple tools in parallel
  const executeTools = useCallback(async (toolCalls: ToolCall[]): Promise<ToolCallResult[]> => {
    setState(prev => ({
      ...prev,
      isExecutingTools: true,
      currentToolCalls: toolCalls,
    }));

    try {
      const results = await Promise.all(toolCalls.map(tc => executeTool(tc)));
      
      setState(prev => ({
        ...prev,
        isExecutingTools: false,
        toolResults: [...prev.toolResults, ...results],
        iterationCount: prev.iterationCount + 1,
      }));

      return results;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isExecutingTools: false,
      }));
      throw error;
    }
  }, [executeTool]);

  // Check if we should continue tool calling
  const shouldContinue = useCallback((): boolean => {
    return state.iterationCount < maxIterations;
  }, [state.iterationCount, maxIterations]);

  // Reset state for new conversation turn
  const reset = useCallback(() => {
    setState({
      isExecutingTools: false,
      currentToolCalls: [],
      toolResults: [],
      iterationCount: 0,
    });
  }, []);

  // Parse tool_calls from non-streaming response
  const parseToolCalls = useCallback((response: Record<string, unknown>): ToolCall[] | null => {
    const choices = response.choices as Array<{
      message?: {
        tool_calls?: ToolCall[];
      };
      finish_reason?: string;
    }>;
    
    if (!choices || choices.length === 0) return null;
    
    const choice = choices[0];
    const toolCalls = choice.message?.tool_calls;
    
    if (toolCalls && toolCalls.length > 0 && choice.finish_reason === 'tool_calls') {
      return toolCalls;
    }
    
    return null;
  }, []);

  // Parse tool_calls from streaming response chunks
  const parseStreamingToolCalls = useCallback((chunks: Array<Record<string, unknown>>): ToolCall[] | null => {
    // Accumulate tool_calls from delta
    const toolCallsMap = new Map<number, ToolCall>();
    let hasToolCalls = false;

    for (const chunk of chunks) {
      const choices = chunk.choices as Array<{
        delta?: {
          tool_calls?: Array<{
            index: number;
            id?: string;
            type?: string;
            function?: {
              name?: string;
              arguments?: string;
            };
          }>;
        };
        finish_reason?: string;
      }>;
      
      if (!choices || choices.length === 0) continue;
      
      const delta = choices[0].delta;
      if (delta?.tool_calls) {
        hasToolCalls = true;
        for (const tc of delta.tool_calls) {
          const existing = toolCallsMap.get(tc.index);
          if (existing) {
            // Append arguments
            if (tc.function?.arguments) {
              existing.function.arguments += tc.function.arguments;
            }
          } else {
            // New tool call
            toolCallsMap.set(tc.index, {
              id: tc.id || `call_${tc.index}`,
              type: 'function',
              function: {
                name: tc.function?.name || '',
                arguments: tc.function?.arguments || '',
              },
            });
          }
        }
      }
      
      // Check finish reason
      if (choices[0].finish_reason === 'tool_calls') {
        hasToolCalls = true;
      }
    }

    if (hasToolCalls && toolCallsMap.size > 0) {
      return Array.from(toolCallsMap.values());
    }
    
    return null;
  }, []);

  return {
    ...state,
    fetchEnabledTools,
    getToolsForRequest,
    executeTools,
    shouldContinue,
    reset,
    parseToolCalls,
    parseStreamingToolCalls,
    enabledTools: enabledToolsRef.current,
  };
}

export default useToolCalling;
