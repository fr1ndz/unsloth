// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/training";
import { listLocalModels } from "@/features/training/api/models-api";
import type { ModelCheckpoints } from "./api/export-api";
import { fetchCheckpoints } from "./api/export-api";

let cachedCheckpoints: ModelCheckpoints[] | null = null;
let cachedLocalModels: LocalModelInfo[] | null = null;

export function getCachedCheckpoints(): ModelCheckpoints[] | null {
  return cachedCheckpoints;
}

export function getCachedLocalModels(): LocalModelInfo[] | null {
  return cachedLocalModels;
}

export async function refreshCheckpoints(): Promise<ModelCheckpoints[]> {
  const response = await fetchCheckpoints();
  cachedCheckpoints = response.models;
  return response.models;
}

export async function refreshLocalModels(): Promise<LocalModelInfo[]> {
  const data = await listLocalModels();
  cachedLocalModels = data;
  return data;
}
