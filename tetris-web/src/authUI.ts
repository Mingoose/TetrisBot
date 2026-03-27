import { supabase } from './supabase';

type Mode = 'signin' | 'signup';

export function setupAuthUI(onSuccess: (userId: string) => void): void {
  let mode: Mode = 'signin';

  const overlay  = document.getElementById('auth-overlay')!;
  const title    = document.getElementById('auth-title')!;
  const emailEl  = document.getElementById('auth-email')! as HTMLInputElement;
  const passEl   = document.getElementById('auth-password')! as HTMLInputElement;
  const errorEl  = document.getElementById('auth-error')!;
  const submitEl = document.getElementById('auth-submit')! as HTMLButtonElement;
  const toggleEl = document.getElementById('auth-toggle-link')!;

  function setMode(m: Mode): void {
    mode = m;
    if (mode === 'signin') {
      title.textContent    = 'SIGN IN';
      submitEl.textContent = 'Sign In';
      toggleEl.textContent = 'Need an account? Sign up';
    } else {
      title.textContent    = 'CREATE ACCOUNT';
      submitEl.textContent = 'Sign Up';
      toggleEl.textContent = 'Already have an account? Sign in';
    }
    errorEl.textContent = '';
  }

  toggleEl.addEventListener('click', () => setMode(mode === 'signin' ? 'signup' : 'signin'));

  submitEl.addEventListener('click', () => submit());
  [emailEl, passEl].forEach(el => el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submit();
  }));

  async function submit(): Promise<void> {
    const email    = emailEl.value.trim();
    const password = passEl.value;
    errorEl.textContent = '';
    submitEl.disabled   = true;

    const { error, data } = mode === 'signin'
      ? await supabase.auth.signInWithPassword({ email, password })
      : await supabase.auth.signUp({ email, password });

    submitEl.disabled = false;

    if (error) {
      errorEl.textContent = error.message;
      return;
    }

    const userId = data.user?.id;
    if (!userId) {
      // Sign-up without email confirmation — shouldn't happen with defaults, but guard anyway
      errorEl.textContent = 'Check your email to confirm your account.';
      return;
    }

    overlay.style.display = 'none';
    onSuccess(userId);
  }

  // Check for an existing session on load — skip the modal if already logged in.
  supabase.auth.getSession().then(({ data: { session } }) => {
    if (session?.user) {
      overlay.style.display = 'none';
      onSuccess(session.user.id);
    }
  });
}
