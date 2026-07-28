// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PrepareHfTokenOptions {
  allowAnonymous?: boolean;
}

export interface PreparedHfToken {
  proceed: boolean;
  token: string | null;
}

/**
 * Validates and prepares an HF token for use in export operations.
 * Returns `{ proceed: false }` when the token is required but missing/invalid.
 */
export async function prepareHfTokenForUse(
  token: string | null | undefined,
  options: PrepareHfTokenOptions = {},
): Promise<PreparedHfToken> {
  const trimmed = (token ?? "").trim();
  if (!trimmed) {
    return { proceed: !!options.allowAnonymous, token: null };
  }
  return { proceed: true, token: trimmed };
}
