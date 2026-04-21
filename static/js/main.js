
"use strict";
const form          = document.getElementById("predict-form");
const textarea      = document.getElementById("message-input");
const charCounter   = document.getElementById("char-counter");
const submitBtn     = document.getElementById("submit-btn");
const resultSection = document.getElementById("result-section");
const resultCard    = document.getElementById("result-card");
const resultIcon    = document.getElementById("result-icon");
const resultLabel   = document.getElementById("result-label");
const resultDesc    = document.getElementById("result-desc");
const confFill      = document.getElementById("confidence-fill");
const confValue     = document.getElementById("confidence-value");

const MAX_CHARS = 500;
textarea.addEventListener("input", () => {
  const len = textarea.value.length;
  charCounter.textContent = `${len} / ${MAX_CHARS}`;
  charCounter.style.color = len > MAX_CHARS * 0.9 ? "#ef4444" : "";
});
document.querySelectorAll(".sample-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.dataset.message;
    textarea.value = text;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
    textarea.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
});
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const message = textarea.value.trim();
  if (!message) {
    showError("Please type a message before clicking Check.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Something went wrong. Please try again.");
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Network error — make sure the Flask server is running.");
  } finally {
    setLoading(false);
  }
});
function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.classList.toggle("loading", on);
}

function showResult({ label, is_spam, confidence }) {
  resultCard.className = "result-card";
  resultSection.classList.remove("visible");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (is_spam) {
        resultCard.classList.add("spam");
        resultIcon.textContent = "🚫";
        resultLabel.textContent = "Spam Detected";
        resultDesc.textContent =
          "This message shows strong signals of spam or phishing content.";
      } else {
        resultCard.classList.add("ham");
        resultIcon.textContent = "✅";
        resultLabel.textContent = "Looks Legitimate";
        resultDesc.textContent =
          "No spam indicators found — this message appears to be genuine.";
      }

      confValue.textContent = `${confidence}%`;
      confFill.style.width = "0%";

      resultSection.classList.add("visible");
      setTimeout(() => {
        confFill.style.width = `${confidence}%`;
      }, 100);
    });
  });
}

function showError(message) {
  resultCard.className = "result-card error";
  resultSection.classList.remove("visible");

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resultIcon.textContent = "⚠️";
      resultLabel.textContent = "Error";
      resultDesc.textContent = message;
      confValue.textContent = "—";
      confFill.style.width = "0%";
      resultSection.classList.add("visible");
    });
  });
}
