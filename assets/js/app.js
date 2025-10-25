let currentUtterance = null;
let speakingButton = null;

function speak(text, lang, buttonElement) {
  if (window.speechSynthesis.speaking && currentUtterance) {
    window.speechSynthesis.cancel();
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    if (speakingButton === buttonElement) {
      currentUtterance = null;
      speakingButton = null;
      return;
    }
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;
  currentUtterance = utterance;
  speakingButton = buttonElement;
  speakingButton.innerHTML = '<i class="fas fa-stop"></i>';

  utterance.onend = () => {
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    currentUtterance = null;
    speakingButton = null;
  };

  utterance.onerror = (event) => {
    console.error("Speech synthesis error:", event.error);
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    currentUtterance = null;
    speakingButton = null;
  };

  window.speechSynthesis.speak(utterance);
}
