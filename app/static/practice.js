(() => {
  const preview = document.getElementById('preview');
  const placeholder = document.getElementById('placeholder');
  const startCamera = document.getElementById('camera-start');
  const startRecord = document.getElementById('record-start');
  const stopRecord = document.getElementById('record-stop');
  const indicator = document.getElementById('recording-indicator');
  const message = document.getElementById('record-message');
  const fileInput = document.getElementById('video-file');
  const fileName = document.getElementById('file-name');
  const form = document.getElementById('upload-form');
  const processing = document.getElementById('processing');
  let stream;
  let recorder;
  let chunks = [];

  startCamera.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      preview.srcObject = stream;
      preview.hidden = false;
      placeholder.hidden = true;
      startCamera.textContent = 'Camera ready';
      startCamera.disabled = true;
      startRecord.disabled = !window.MediaRecorder;
      message.textContent = window.MediaRecorder ? 'When ready, record a short signal and hold it clearly.' : 'Recording is unavailable in this browser. Choose a video file instead.';
    } catch (error) {
      message.textContent = 'Camera access was unavailable. Check browser permissions or choose a video file.';
    }
  });

  startRecord.addEventListener('click', () => {
    if (!stream) return;
    const mimeType = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) {
      message.textContent = 'This browser cannot create a supported recording. Choose a video file instead.';
      return;
    }
    chunks = [];
    recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
      const file = new File([blob], `practice-${Date.now()}.${ext}`, { type: mimeType });
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
      fileName.textContent = file.name;
      message.textContent = 'Recording ready. Submit it to see your practice summary.';
      preview.srcObject = null;
      preview.src = URL.createObjectURL(blob);
      preview.controls = true;
      stream.getTracks().forEach(track => track.stop());
      stream = null;
      startCamera.disabled = false;
      startCamera.textContent = 'Record again';
      startRecord.disabled = true;
      stopRecord.disabled = true;
    };
    recorder.start(500);
    startRecord.disabled = true;
    stopRecord.disabled = false;
    indicator.hidden = false;
    message.textContent = 'Recording… make the signal clearly, then stop.';
  });

  stopRecord.addEventListener('click', () => {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    indicator.hidden = true;
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
      fileName.textContent = file.name;
      if (preview.srcObject) {
        preview.srcObject.getTracks().forEach(track => track.stop());
        preview.srcObject = null;
      }
      preview.hidden = false;
      preview.src = URL.createObjectURL(file);
      preview.controls = true;
      placeholder.hidden = true;
      message.textContent = 'Video selected. Submit when you are ready.';
    }
  });

  form.addEventListener('submit', () => {
    processing.hidden = false;
    form.querySelector('button[type="submit"]').disabled = true;
  });
  window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
})();
