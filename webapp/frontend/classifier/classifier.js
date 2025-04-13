const uploadInput = document.getElementById("upload");
const uploadBox = document.getElementById("uploadBox");
const previewCard = document.getElementById("previewCard");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const removeBtn = document.getElementById("removeFile");
const analyzeBtn = document.querySelector('.analyze-btn');
const classifyResult = document.getElementById('results-container');

const statusDiv = document.getElementById('status');
const classificationResult = document.getElementById('classification-result');
const gradcamVisualization = document.getElementById('gradcam-visualization');

// Handle image preview
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    fileName.textContent = file.name;
    previewImage.src = URL.createObjectURL(file);
    // previewImage.style.height = "400px";
    previewImage.style.verticalAlign = "middle";
    // previewImage.style.marginRight = "8px";


    uploadBox.style.display = "none";
    previewCard.style.display = "block";
  }
});

removeBtn.addEventListener("click", () => {
  uploadInput.value = "";
  previewCard.style.display = "none";
  uploadBox.style.display = "flex";
  classifyResult.innerHTML = "";
});


async function sendImage() {
  if (!uploadInput.files.length) {
    alert('Please select an image.');
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner"></span> Processing...';

  const formData = new FormData();
  formData.append('file', uploadInput.files[0]);



  try {
    const response = await fetch('https://adapt-api-h0ym.onrender.com/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      alert('Prediction failed!');
      return;
    }

    const data = await response.json();

    let textColor;
    let backgroundColor;
    switch (data.class_code) {
      case 'AD':
        textColor = 'red';
        backgroundColor = 'rgba(255, 0, 0, 0.1)';
        break;
      case 'CN':
        textColor = 'green';
        backgroundColor = 'rgba(0, 128, 0, 0.1)';
        break;
      case 'EMCI':
        textColor = 'goldenrod'; // Yellowish
        backgroundColor = 'rgba(218, 165, 32, 0.1)';
        break;
      case 'LMCI':
        textColor = 'mediumorchid';
        backgroundColor = 'rgba(186, 85, 211, 0.1)';
        break;
      default:
        textColor = 'black';
        backgroundColor = 'transparent';
    }

    classifyResult.innerHTML = "";

    const statusDiv = document.createElement("div");
    statusDiv.id = "status";

    statusDiv.innerHTML = `
      <img src="../assets/success1.png" alt="Success"
         style="height: 20px; vertical-align: middle; margin-right: 8px;">
      <span style="font-weight: 600; font-size: 1.2rem;">
        Analysis Complete
      </span>
    `;

    classifyResult.appendChild(statusDiv);

    const classificationResult = document.createElement('div');
    classificationResult.id = "classification-result";

    const span = document.createElement('span');
    span.textContent = "Classification Result";
    classificationResult.appendChild(span);

    const resultWindow = document.createElement('div');
    const div1 = document.createElement('div');
    const div2 = document.createElement('div');

    div1.innerHTML = `
      <span>Detected Stage:</span>
      <span  style="padding: 0 8px; color: ${textColor}; background-color: ${backgroundColor};">${data.predicted_class} (${data.class_code})</span>
    `;
    div1.classList.add('predicted-class');

    div2.innerHTML = `
      <span>Confidence Score:</span>
    `;

    const confidenceValues = document.createElement('div');

    const confidenceBar = document.createElement('div');
    confidenceBar.classList.add('confidence-bar');

    const confidenceFill = document.createElement('span');
    confidenceFill.id = 'confidence-fill';

    confidenceFill.style.width = `${data.confidence}%`

    if (data.confidence >= 80) {
      confidenceFill.style.backgroundColor = "#4caf50"; // Green
    } else if (data.confidence >= 60) {
      confidenceFill.style.backgroundColor = "#ffca28"; // Amber
    } else {
      confidenceFill.style.backgroundColor = "#f44336"; // Red
    }

    confidenceBar.appendChild(confidenceFill);

    confidenceValues.appendChild(confidenceBar);

    const confidencePercent = document.createElement('div');
    confidencePercent.textContent = `${data.confidence}%`;
    confidencePercent.style.color = 'black';
    confidencePercent.style.fontSize = "12px";

    confidenceValues.appendChild(confidencePercent);

    confidenceValues.classList.add('confidence-values');

    div2.appendChild(confidenceValues);

    div2.classList.add('confidence');

    resultWindow.appendChild(div1);
    resultWindow.appendChild(div2);

    resultWindow.classList.add('result-window');

    classificationResult.appendChild(resultWindow);
    classificationResult.classList.add('classification-result');

    classifyResult.appendChild(classificationResult);

    const gradcamVisualization = document.createElement('div');
    gradcamVisualization.id = "gradcam-visualization";

    const gradcamTitle = document.createElement('div');
    const gradcamTitleIcon = document.createElement('img');
    gradcamTitleIcon.src = "../assets/gradcam.svg";
    gradcamTitleIcon.style.height = "20px";
    gradcamTitleIcon.style.verticalAlign = "middle";
    gradcamTitleIcon.style.marginRight = "8px";

    const gradcamText = document.createElement('span');
    gradcamText.textContent = 'GradCAM Visualisation';

    gradcamText.classList.add('gradcam-title-text');

    gradcamTitle.appendChild(gradcamTitleIcon);
    gradcamTitle.appendChild(gradcamText);

    gradcamVisualization.appendChild(gradcamTitle);

    const info1 = document.createElement('span');
    info1.textContent = "The highlighted areas show regions of the brain that most influenced the classification decision."
    info1.style.color = "oklch(37.2% 0.044 257.287)";

    gradcamVisualization.appendChild(info1);

    const gradcamArea = document.createElement('div');
    const gradCamImage = document.createElement('img');
    gradCamImage.src = `data:image/png;base64,${data.gradcam}`;

    gradCamImage.classList.add('gradcam-image');

    gradcamArea.appendChild(gradCamImage);

    gradcamArea.classList.add('gradcam-area');

    gradcamVisualization.appendChild(gradcamArea);

    const info2 = document.createElement('span');
    info2.textContent = "Red areas indicate regions with the strongest influence on the classification."
    info2.style.color = "oklch(55.4% 0.046 257.417)"

    gradcamVisualization.appendChild(info2);

    gradcamVisualization.classList.add('gradcam-visualization');

    classifyResult.appendChild(gradcamVisualization);

  } catch (error) {
    console.error('Error parsing JSON:', error.message);
    statusDiv.innerHTML = `
      <img src="../assets/failure1.png" alt="Failed"
         style="height: 20px; vertical-align: middle; margin-right: 8px;">
      <span style="font-weight: 600; font-size: 1.2rem;">
        Analysis Failed
      </span>
    `;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `
      <img id="btn-logo" src="../assets/upload-icon2.svg" alt="Upload Icon" />
      <span>Analyze Brain Scan</span>
    `
  }
}