/* Static script envelopes work over file:// as well as HTTP. No network API. */
(() => {
  const cache = new Map(), waiting = new Map();
  function trim() {
    const chunks = [...cache.keys()].filter(k => !k.endsWith('/meta'));
    while (chunks.length > 6) cache.delete(chunks.shift());
  }
  window.PAULA_DATA = {
    async receive(key, base64) {
      const pending = waiting.get(key);
      if (!pending) return;
      try {
        const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
        const text = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
        const payload = JSON.parse(text);
        cache.set(key, payload); trim(); pending.resolve(payload);
      } catch (error) { pending.reject(error); }
      finally { clearTimeout(pending.timer); pending.script.remove(); waiting.delete(key); }
    },
    get(key) {
      if (!/^[a-z-]+\/(meta|\d+)$/.test(key)) return Promise.reject(new Error('Invalid recording key'));
      if (!window.DecompressionStream) return Promise.reject(new Error('This browser cannot decompress the recording. Use a current Chrome, Edge, or Safari.'));
      if (cache.has(key)) { const data = cache.get(key); cache.delete(key); cache.set(key, data); return Promise.resolve(data); }
      if (waiting.has(key)) return waiting.get(key).promise;
      const script = document.createElement('script');
      const [run, chunk] = key.split('/');
      script.src = `data/${run}/${chunk === 'meta' ? 'meta' : 'trial-'+chunk}.js`;
      let resolve, reject;
      const promise = new Promise((a, b) => { resolve = a; reject = b; });
      const fail = () => { clearTimeout(waiting.get(key)?.timer); waiting.delete(key); script.remove(); reject(new Error(`Could not load ${key}. Reload, or regenerate the offline replay data.`)); };
      waiting.set(key, {promise, resolve, reject, script, timer:setTimeout(fail, 20000)});
      script.onerror = fail;
      document.head.append(script);
      return promise;
    },
    cacheSize: () => ({chunks:[...cache.keys()].filter(k => !k.endsWith('/meta')).length, pending:waiting.size})
  };
})();
