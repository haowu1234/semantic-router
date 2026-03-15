import type {
  NLAuthoringCapabilities,
  NLSchemaManifest,
  NLSessionCreateRequest,
  NLSessionCreateResponse,
  NLTurnRequest,
  NLTurnResponse,
} from '../types/nl'

const API_BASE = '/api/builder/nl'

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(text || `HTTP ${response.status}: ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function getNLAuthoringCapabilities(): Promise<NLAuthoringCapabilities> {
  const response = await fetch(`${API_BASE}/capabilities`)
  return handleResponse<NLAuthoringCapabilities>(response)
}

export async function getNLAuthoringSchema(): Promise<NLSchemaManifest> {
  const response = await fetch(`${API_BASE}/schema`)
  return handleResponse<NLSchemaManifest>(response)
}

export async function createNLAuthoringSession(
  request: NLSessionCreateRequest,
): Promise<NLSessionCreateResponse> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  return handleResponse<NLSessionCreateResponse>(response)
}

export async function createNLAuthoringTurn(
  sessionId: string,
  request: NLTurnRequest,
): Promise<NLTurnResponse> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/turns`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  return handleResponse<NLTurnResponse>(response)
}
