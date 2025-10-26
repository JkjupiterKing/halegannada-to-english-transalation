const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const photo = document.getElementById("photo");
const captureBtn = document.getElementById("captureBtn");
const extractedText = document.getElementById("extractedText");
const translatedText = document.getElementById("translatedText");
const loader = document.getElementById("loader");

// Get access to the camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
  } catch (err) {
    console.error("Error accessing the camera: ", err);
    alert("Could not access the camera. Please allow camera access.");
  }
}

// Capture a photo
captureBtn.addEventListener("click", async () => {
  extractedText.value = "";
  translatedText.value = "";
  loader.style.display = "block";

  const context = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/png");
  photo.setAttribute("src", dataUrl);
  photo.style.display = "block";

  try {
    const {
      data: { text },
    } = await Tesseract.recognize(dataUrl, "kan", {
      logger: (m) => console.log(m),
    });

    extractedText.value = text;
    if (text.trim()) {
      await translateText(text);
    }
  } catch (error) {
    console.error("OCR Error:", error);
    alert("Failed to recognize text. Please try again.");
  } finally {
    loader.style.display = "none";
  }
});

// Translate text
async function translateText(text) {
  try {
    const response = await fetch("/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
        source: "dictionary", // Using dictionary for consistency, though server uses API for English
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    if (data.english) {
      translatedText.value = data.english;
    } else {
      translatedText.value = "Translation not found.";
    }
  } catch (error) {
    console.error("Translation Error:", error);
    translatedText.value = "Failed to get translation.";
  }
}

// Start the camera when the page loads
window.addEventListener("load", startCamera);