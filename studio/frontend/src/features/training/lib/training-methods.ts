// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetFormat, TrainingMethod } from "@/types/training";

const BACKEND_TRAINING_TYPE: Record<TrainingMethod, string> = {
  qlora: "LoRA/QLoRA",
  lora: "LoRA/QLoRA",
  full: "Full Finetuning",
  cpt: "Continued Pretraining",
  "bonsai-lora": "Bonsai LoRA",
  "1bit-lora": "1-bit LoRA",
  "1bit-qlora": "1-bit QLoRA",
  "1bit-loftq": "1-bit LOFTQ",
  "1bit-full": "1-bit Full Finetuning",
};

const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full",
  cpt: "CPT",
  "bonsai-lora": "Bonsai LoRA",
  "1bit-lora": "1-bit LoRA",
  "1bit-qlora": "1-bit QLoRA",
  "1bit-loftq": "1-bit LOFTQ",
  "1bit-full": "1-bit Full",
};

export function toBackendTrainingType(trainingMethod: TrainingMethod): string {
  return BACKEND_TRAINING_TYPE[trainingMethod];
}

export function getTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
): string {
  if (Object.prototype.hasOwnProperty.call(TRAINING_METHOD_LABELS, trainingMethod)) {
    return TRAINING_METHOD_LABELS[trainingMethod as TrainingMethod];
  }
  return TRAINING_METHOD_LABELS.full;
}

export function parseBackendTrainingMethod(
  trainingType: unknown,
  loadIn4Bit: unknown,
): TrainingMethod {
  if (trainingType === "Continued Pretraining") return "cpt";
  if (trainingType === "Bonsai LoRA") return "bonsai-lora";
  if (trainingType === "1-bit LoRA") return "1bit-lora";
  if (trainingType === "1-bit QLoRA") return "1bit-qlora";
  if (trainingType === "1-bit LOFTQ") return "1bit-loftq";
  if (trainingType === "1-bit Full Finetuning") return "1bit-full";
  if (trainingType === "LoRA/QLoRA") {
    return loadIn4Bit ? "qlora" : "lora";
  }
  return "full";
}

export function isRawTextDatasetFormat(
  datasetFormat: DatasetFormat,
): boolean {
  return datasetFormat === "raw";
}
