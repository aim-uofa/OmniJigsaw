document.documentElement.classList.add('js');

const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');

function closeMenu() {
  navigation.classList.remove('is-open');
  menuButton.setAttribute('aria-expanded', 'false');
}

menuButton.addEventListener('click', () => {
  const expanded = menuButton.getAttribute('aria-expanded') !== 'true';
  navigation.classList.toggle('is-open', expanded);
  menuButton.setAttribute('aria-expanded', String(expanded));
});
navigation.addEventListener('click', event => {
  if (event.target.closest('a')) closeMenu();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && navigation.classList.contains('is-open')) {
    closeMenu();
    menuButton.focus();
  }
});

const selection = { backbone: '30b', domain: 'video', mode: 'audio' };
const controls = document.querySelector('.result-controls');

function updateResults() {
  controls.querySelectorAll('[data-filter]').forEach(button => {
    button.setAttribute('aria-pressed', String(selection[button.dataset.filter] === button.dataset.value));
  });
  document.querySelector('#audio-controls').hidden = selection.domain !== 'video';
  document.querySelectorAll('.result-panel').forEach(panel => {
    panel.hidden = panel.dataset.backbone !== selection.backbone ||
      panel.dataset.domain !== selection.domain ||
      (selection.domain === 'video' && panel.dataset.mode !== selection.mode);
  });
}

controls.addEventListener('click', event => {
  const button = event.target.closest('[data-filter]');
  if (!button) return;
  selection[button.dataset.filter] = button.dataset.value;
  updateResults();
});
updateResults();

let citationFeedbackTimer;
document.querySelector('#copy-citation').addEventListener('click', async event => {
  const button = event.currentTarget;
  const citation = document.querySelector('#citation-block').textContent.trim();
  const feedback = document.querySelector('#copy-feedback');
  clearTimeout(citationFeedbackTimer);
  try {
    await navigator.clipboard.writeText(citation);
    button.textContent = 'Copied!';
    feedback.textContent = 'Citation copied to clipboard.';
    citationFeedbackTimer = setTimeout(() => {
      button.textContent = 'Copy Citation';
      feedback.textContent = '';
    }, 2500);
  } catch {
    button.textContent = 'Copy Citation';
    feedback.textContent = 'Please select and copy the citation above.';
  }
});
