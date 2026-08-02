const grid = document.getElementById('layer-grid');
const controls = document.querySelector('.controls-right');
const label = document.createElement('label');
const toggle = document.createElement('input');

label.className = 'magnify-toggle';
toggle.type = 'checkbox';
toggle.checked = true;
label.append(toggle, 'Magnify');
controls.prepend(label);
grid.classList.add('magnify-enabled');

toggle.addEventListener('change', () => {
  grid.classList.toggle('magnify-enabled', toggle.checked);
  if (!toggle.checked) resetAll();
});

grid.addEventListener('mousemove', (event) => {
  if (!toggle.checked || !isMagnifiable(event.target)) return;
  const image = event.target;
  const bounds = image.parentElement.getBoundingClientRect();
  const x = 100 * (event.clientX - bounds.left) / bounds.width;
  const y = 100 * (event.clientY - bounds.top) / bounds.height;
  image.style.transformOrigin = `${x}% ${y}%`;
  image.style.transform = 'scale(3)';
});

grid.addEventListener('mouseout', (event) => {
  if (isMagnifiable(event.target)) reset(event.target);
});

function isMagnifiable(element) {
  return element instanceof HTMLImageElement
    || element instanceof HTMLCanvasElement;
}

function reset(image) {
  image.style.transform = '';
  image.style.transformOrigin = '';
}

function resetAll() {
  grid.querySelectorAll('img, canvas').forEach(reset);
}
