(() => {
  const video = document.querySelector('[data-match-camera]');
  if (!video) return;
  const startButton = document.querySelector('[data-camera-start]');
  const stopButton = document.querySelector('[data-camera-stop]');
  const status = document.querySelector('[data-camera-status]');
  const reading = document.querySelector('[data-detector-reading]');
  let stream = null;
  let timer = null;
  let inFlight = false;
  const clientId = `live-${crypto.randomUUID ? crypto.randomUUID() : Date.now()}`;

  const stop = () => {
    if (timer) window.clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    video.hidden = true;
    startButton.disabled = false;
    stopButton.disabled = true;
  };

  stopButton.addEventListener('click', () => {
    stop();
    status.textContent = 'Camera stopped.';
  });

  startButton.addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      status.textContent = 'Camera access is unavailable. Use HTTPS or localhost, or enter the event manually.';
      return;
    }
    startButton.disabled = true;
    status.textContent = 'Requesting camera access…';
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      video.srcObject = stream;
      video.hidden = false;
      await video.play();
      stopButton.disabled = false;
      status.textContent = 'Camera on. Hold one signal clearly in view. Frames are processed and discarded.';
      timer = window.setInterval(captureFrame, 1250);
    } catch (error) {
      stop();
      status.textContent = error.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access in the browser settings or enter events manually.'
        : 'Could not start the camera. Check that it is connected and not in use by another app.';
      startButton.disabled = false;
    }
  });

  async function captureFrame() {
    if (inFlight || !stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return;
    inFlight = true;
    try {
      const canvas = document.createElement('canvas');
      const ratio = Math.min(1, 720 / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * ratio);
      canvas.height = Math.round(video.videoHeight * ratio);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.72));
      if (!blob) return;
      const body = new FormData();
      body.append('csrf_token', video.dataset.csrf);
      body.append('client_id', clientId);
      body.append('frame', blob, 'frame.jpg');
      const response = await fetch(startButton.dataset.detectUrl, { method: 'POST', body, credentials: 'same-origin' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Frame analysis failed.');
      reading.textContent = `${result.label} · ${(result.confidence * 100).toFixed(0)}% confidence`;
      reading.classList.toggle('candidate', result.candidate);
      if (result.event_created) {
        stop();
        status.textContent = 'Stable signal proposal added. Refresh to review and score it.';
      }
    } catch (error) {
      status.textContent = error.message || 'Camera analysis failed.';
      if (error.message && /complete|missing|server/i.test(error.message)) stop();
    } finally {
      inFlight = false;
    }
  }

  window.addEventListener('pagehide', stop, { once: true });
})();
