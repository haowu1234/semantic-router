import { GENERATED_NL_SCHEMA_MANIFEST } from "./generatedNlSchemaManifest";

export interface CatalogFieldSchema {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "string[]" | "number[]" | "select" | "json";
  options?: string[];
  required?: boolean;
  placeholder?: string;
  description?: string;
}

interface CatalogEntry {
  typeName: string;
  description: string;
  fields: CatalogFieldSchema[];
}

export type SignalCatalogType =
  (typeof GENERATED_NL_SCHEMA_MANIFEST.signals)[number]["typeName"];
export type PluginCatalogType =
  (typeof GENERATED_NL_SCHEMA_MANIFEST.plugins)[number]["typeName"];
export type AlgorithmCatalogType =
  (typeof GENERATED_NL_SCHEMA_MANIFEST.algorithms)[number]["typeName"];
export type GeneratedBackendCatalogType =
  (typeof GENERATED_NL_SCHEMA_MANIFEST.backends)[number]["typeName"];
export type BackendCatalogType =
  | GeneratedBackendCatalogType
  | "vector_store"
  | "image_gen_backend";

const LEGACY_BACKEND_ENTRIES: CatalogEntry[] = [
  {
    typeName: "vector_store",
    description: "Vector store for RAG retrieval.",
    fields: [],
  },
  {
    typeName: "image_gen_backend",
    description: "Image generation backend (DALL-E, SD, etc.).",
    fields: [],
  },
];

function normalizeEntries(
  entries: ReadonlyArray<{
    typeName: string;
    description: string;
    fields?: ReadonlyArray<{
      key: string;
      label: string;
      type: CatalogFieldSchema["type"];
      options?: ReadonlyArray<string>;
      required?: boolean;
      placeholder?: string;
      description?: string;
    }>;
  }>,
): CatalogEntry[] {
  return entries.map((entry) => ({
    typeName: entry.typeName,
    description: entry.description,
    fields: (entry.fields ?? []).map((field) => ({
      key: field.key,
      label: field.label,
      type: field.type,
      options: field.options ? [...field.options] : undefined,
      required: field.required,
      placeholder: field.placeholder,
      description: field.description,
    })),
  }));
}

function buildDescriptionMap(entries: CatalogEntry[]): Record<string, string> {
  return Object.fromEntries(entries.map((entry) => [entry.typeName, entry.description]));
}

function buildFieldMap(entries: CatalogEntry[]): Record<string, CatalogFieldSchema[]> {
  return Object.fromEntries(entries.map((entry) => [entry.typeName, entry.fields]));
}

const signalEntries = normalizeEntries(GENERATED_NL_SCHEMA_MANIFEST.signals);
const pluginEntries = normalizeEntries(GENERATED_NL_SCHEMA_MANIFEST.plugins);
const algorithmEntries = normalizeEntries(GENERATED_NL_SCHEMA_MANIFEST.algorithms);
const backendEntries = [
  ...normalizeEntries(GENERATED_NL_SCHEMA_MANIFEST.backends),
  ...LEGACY_BACKEND_ENTRIES,
];

export const SIGNAL_TYPES = signalEntries.map((entry) => entry.typeName) as SignalCatalogType[];
export const SIGNAL_DESCRIPTIONS = buildDescriptionMap(signalEntries);
const SIGNAL_FIELD_MAP = buildFieldMap(signalEntries);

export const PLUGIN_TYPES = pluginEntries.map((entry) => entry.typeName) as PluginCatalogType[];
export const PLUGIN_DESCRIPTIONS = buildDescriptionMap(pluginEntries);
const PLUGIN_FIELD_MAP = buildFieldMap(pluginEntries);

export const ALGORITHM_TYPES = algorithmEntries.map((entry) => entry.typeName) as AlgorithmCatalogType[];
export const ALGORITHM_DESCRIPTIONS = buildDescriptionMap(algorithmEntries);
const ALGORITHM_FIELD_MAP = buildFieldMap(algorithmEntries);

export const BACKEND_TYPES = backendEntries.map((entry) => entry.typeName) as BackendCatalogType[];
export const BACKEND_DESCRIPTIONS = buildDescriptionMap(backendEntries);
const BACKEND_FIELD_MAP = buildFieldMap(backendEntries);

export function getSignalFieldSchema(signalType: string): CatalogFieldSchema[] {
  return SIGNAL_FIELD_MAP[signalType]?.map((field) => ({ ...field, options: field.options ? [...field.options] : undefined })) ?? [
    { key: "description", label: "Description", type: "string" },
  ];
}

export function getPluginFieldSchema(pluginType: string): CatalogFieldSchema[] {
  return PLUGIN_FIELD_MAP[pluginType]?.map((field) => ({ ...field, options: field.options ? [...field.options] : undefined })) ?? [
    { key: "enabled", label: "Enabled", type: "boolean" },
  ];
}

export function getAlgorithmFieldSchema(algoType: string): CatalogFieldSchema[] {
  return ALGORITHM_FIELD_MAP[algoType]?.map((field) => ({ ...field, options: field.options ? [...field.options] : undefined })) ?? [];
}

export function getBackendFieldSchema(backendType: string): CatalogFieldSchema[] {
  return BACKEND_FIELD_MAP[backendType]?.map((field) => ({ ...field, options: field.options ? [...field.options] : undefined })) ?? [];
}
