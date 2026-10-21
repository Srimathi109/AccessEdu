(() => {
  const preview = document.getElementById('preview');
  const placeholder = document.getElementById('placeholder');
  const countdown = document.getElementById('countdown');
  const message = document.getElementById('record-message');
  const fileInput = document.getElementById('photo-file');
  const canvas = document.getElementById('capture-canvas');
  const form = document.getElementById('upload-form');
  const processing = document.getElementById('processing');
  let stream;

  function stopCamera() {
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    preview.srcObject = null;
  }

  function submitCapturedPhoto(blob) {
    const file = new File([blob], `practice-${Date.now()}.jpg`, { type: 'image/jpeg' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    stopCamera();
    preview.hidden = true;
    countdown.hidden = true;
    processing.hidden = false;
    message.textContent = 'Photo captured. Sending it for analysis…';
    form.requestSubmit();
  }

  async function startAutomaticCapture() {
    if (!navigator.mediaDevices?.getUserMedia) {
      message.textContent = 'This browser cannot access a camera. Open the app in a browser that supports camera access.';
      placeholder.querySelector('span').textContent = 'Camera is unavailable';
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      preview.srcObject = stream;
      preview.hidden = false;
      placeholder.hidden = true;
      await preview.play();

      let remaining = 5;
      countdown.textContent = remaining;
      countdown.hidden = false;
      message.textContent = 'Hold your signal steady. Capturing automatically…';

      const timer = window.setInterval(() => {
        remaining -= 1;
        countdown.textContent = remaining > 0 ? remaining : '📷';
        if (remaining > 0) return;
        window.clearInterval(timer);
        canvas.width = preview.videoWidth;
        canvas.height = preview.videoHeight;
        if (!canvas.width || !canvas.height) {
          stopCamera();
          countdown.hidden = true;
          message.textContent = 'The camera did not provide an image. Reload the page and allow camera access.';
          return;
        }
        canvas.getContext('2d').drawImage(preview, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          if (!blob) {
            stopCamera();
            countdown.hidden = true;
            message.textContent = 'The photo could not be captured. Reload the page to try again.';
            return;
          }
          submitCapturedPhoto(blob);
        }, 'image/jpeg', 0.92);
      }, 1000);
    } catch (error) {
      stopCamera();
      placeholder.querySelector('span').textContent = 'Camera access is needed';
      message.textContent = 'Allow camera access in your browser, then reload this page. No recording or manual capture is required.';
    }
  }

  form.addEventListener('submit', () => {
    processing.hidden = false;
    message.textContent = 'Analyzing your photo…';
  });
  window.addEventListener('beforeunload', stopCamera);
  startAutomaticCapture();
})();
