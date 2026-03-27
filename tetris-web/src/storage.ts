import { Settings, DEFAULT_KEYBINDINGS, DEFAULT_DAS, DEFAULT_ARR, DEFAULT_SONIC_DROP, DEFAULT_BOT_PPS } from './settings';
import { supabase } from './supabase';

// Placeholder until auth lands in Stage 5. Callers will switch to the real UUID.
export const GUEST_USER_ID = 'guest';

// ---- Adapter interface ----
export interface StorageAdapter {
  save(userId: string, settings: Settings): Promise<void>;
  load(userId: string): Promise<Settings | null>;
}

// ---- Supabase implementation ----
class SupabaseSettingsAdapter implements StorageAdapter {
  // Skip until real auth is wired up (Stage 5). Once auth lands, callers pass
  // the actual UUID from supabase.auth.getUser() and these calls go through.
  private isRealUserId(userId: string): boolean {
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(userId);
  }

  async save(userId: string, settings: Settings): Promise<void> {
    if (!this.isRealUserId(userId)) return;
    await supabase
      .from('user_settings')
      .upsert({ user_id: userId, settings }, { onConflict: 'user_id' });
  }

  async load(userId: string): Promise<Settings | null> {
    if (!this.isRealUserId(userId)) return null;
    const { data } = await supabase
      .from('user_settings')
      .select('settings')
      .eq('user_id', userId)
      .single();
    return (data?.settings as Settings) ?? null;
  }
}

const adapter: StorageAdapter = new SupabaseSettingsAdapter();

export async function saveSettings(userId: string, settings: Settings): Promise<void> {
  await adapter.save(userId, settings);
}

export async function loadSettings(userId: string): Promise<Settings> {
  const saved = await adapter.load(userId);
  // Merge with defaults so new keybindings added in future versions get their defaults
  return {
    keybindings: { ...DEFAULT_KEYBINDINGS, ...saved?.keybindings },
    das: saved?.das ?? DEFAULT_DAS,
    arr: saved?.arr ?? DEFAULT_ARR,
    sonicDrop: saved?.sonicDrop ?? DEFAULT_SONIC_DROP,
    botPps: Math.max(0.1, saved?.botPps ?? DEFAULT_BOT_PPS),
  };
}
