// Text-to-speech functionality
let currentUtterance = null;
let speakingButton = null;

function speak(text, lang, buttonElement) {
  if (window.speechSynthesis.speaking && currentUtterance) {
    window.speechSynthesis.cancel();
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
    if (speakingButton === buttonElement) {
      // If the same button is clicked again
      currentUtterance = null;
      speakingButton = null;
      return; // Stop here, don't start new speech
    }
  }

  // If a different button is clicked or no speech is active
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;
  currentUtterance = utterance;
  speakingButton = buttonElement;
  speakingButton.innerHTML = '<i class="fas fa-stop"></i>'; // Change to stop icon

  utterance.onend = () => {
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>'; // Revert icon on end
    }
    currentUtterance = null;
    speakingButton = null;
  };

  utterance.onerror = (event) => {
    console.error("Speech synthesis error:", event.error);
    if (speakingButton) {
      speakingButton.innerHTML = '<i class="fas fa-volume-up"></i>'; // Revert icon on error
    }
    currentUtterance = null;
    speakingButton = null;
  };

  window.speechSynthesis.speak(utterance);
}