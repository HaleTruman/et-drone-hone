'use strict';

// Local illustration and saved images only: no inference, fetch, or server.
const svgNS = 'http://www.w3.org/2000/svg';
const poseControls = [...document.querySelectorAll('[data-pose]')];
let pose = { ...GuideProjection.defaultPose };
let projectionFrame = 0;

function updateProjection() {
  projectionFrame = 0;
  const result = GuideProjection.project(pose);
  const faces = result.faces.map(face => {
    const polygon = document.createElementNS(svgNS, 'polygon');
    polygon.setAttribute('points', face.points.map(point => point.join(',')).join(' '));
    polygon.setAttribute('fill', face.color);
    return polygon;
  });
  document.querySelector('#projected-rails').replaceChildren(...faces);
  const [u, v] = result.center;
  const center = document.querySelector('#projection-center');
  center.setAttribute('cx', u);
  center.setAttribute('cy', v);
  const label = document.querySelector('#projection-center-label');
  label.setAttribute('x', u + 9);
  label.setAttribute('y', v - 6);
  document.querySelector('#projection-count').textContent =
    'Estimated instance pixels: ' + result.pixelCount.toLocaleString('en-US');
  document.querySelector('#projection-description').textContent =
    `Illustrative gate at X ${pose.x.toFixed(1)}, Y ${pose.y.toFixed(1)}, Z ${pose.z.toFixed(1)} meters; ` +
    `roll ${pose.roll} degrees about Z, pitch ${pose.pitch} degrees about X, yaw ${pose.yaw} degrees about Y. ` +
    `${result.pixelCount} image pixels cover the projected rails.`;
}
function syncPoseControls() {
  poseControls.forEach(input => {
    const key = input.dataset.pose;
    input.value = pose[key];
    const value = ['x', 'y', 'z'].includes(key) ? pose[key].toFixed(1) + ' m' : pose[key] + '°';
    document.querySelector('#' + key + '-value').value = value;
    input.setAttribute('aria-valuetext', value);
  });
}
poseControls.forEach(input => input.addEventListener('input', () => {
  pose[input.dataset.pose] = Number(input.value);
  syncPoseControls();
  if (!projectionFrame) projectionFrame = requestAnimationFrame(updateProjection);
}));
document.querySelector('#randomize-projection').addEventListener('click', () => {
  pose = GuideProjection.randomPose();
  syncPoseControls();
  if (projectionFrame) cancelAnimationFrame(projectionFrame);
  updateProjection();
});
syncPoseControls();
updateProjection();

const ringPath = (x,y,w,r) => `M${x} ${y}h${w}v${w}h-${w}Z M${x+r} ${y+r}v${w-2*r}h${w-2*r}v-${w-2*r}Z`;
let caseId = 0;
document.querySelectorAll('svg[data-case]').forEach(svg=>{
  const kind = svg.dataset.case, id='case-'+(++caseId);
  let visual = `<rect x="8" y="9" width="214" height="152" rx="3" fill="#f0f3e9"/>`;
  const inferred = `<path d="${ringPath(61,32,110,25)}" fill="none" stroke="#aa762c" stroke-width="2" stroke-dasharray="5 4"/>`;
  if(kind==='full') visual+=`<path d="${ringPath(61,32,110,25)}" fill="#286953" fill-rule="evenodd"/>`;
  if(kind==='c') visual+=inferred+`<path d="M61 32H171V57H86V117H171V142H61Z" fill="#286953"/>`;
  if(kind==='clipped') visual+=`<defs><pattern id="${id}-h" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7L7 0" stroke="#c5cebd" stroke-width="1"/></pattern><clipPath id="${id}-clip"><rect x="8" y="9" width="148" height="152"/></clipPath></defs><rect x="156" y="9" width="66" height="152" fill="url(#${id}-h)"/><path d="${ringPath(82,32,110,25)}" fill="none" stroke="#aa762c" stroke-width="2" stroke-dasharray="5 4"/><path d="${ringPath(82,32,110,25)}" fill="#286953" fill-rule="evenodd" clip-path="url(#${id}-clip)"/><path d="M156 9V161" stroke="#73806c" stroke-width="2"/>`;
  if(kind==='overlap') visual+=`<path d="${ringPath(34,26,94,22)}" fill="#286953" fill-rule="evenodd"/><path d="${ringPath(98,56,94,22)}" fill="#286953" fill-rule="evenodd"/><path d="M98 56H192V150H98V120" stroke="#afd1b4" fill="none" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  if(kind==='sparse') visual+=`<path d="${ringPath(35,32,105,24)}" fill="none" stroke="#aa762c" stroke-width="2" stroke-dasharray="5 4"/><rect x="35" y="32" width="105" height="24" fill="#286953"/><path d="${ringPath(182,93,13,3)}" fill="#286953" fill-rule="evenodd"/>`;
  if(kind==='missing') visual+=inferred;
  svg.insertAdjacentHTML('beforeend',visual);
});

const steps = [
  {kicker:'STEP 1 / INPUT EVIDENCE',title:'Keep the mask as evidence.',body:'Start with the target pixels that survived detection. A connected patch may contain several gates, and one gate may appear in separate pieces.',detail:'Visible openings and nearby fragments offer starting clues. Missing rails remain unknown.',takeaway:'Keep what was seen separate from what is inferred.'},
  {kicker:'STEP 2 / INITIAL ESTIMATES',title:'Try several positions and angles.',body:'Use the size and arrangement of the visible clues to place a few candidate gates in front of the camera.',detail:'Each candidate uses the same known dimensions. Different starting poses help explore different explanations.',takeaway:'A starting estimate is a possibility to test.'},
  {kicker:'STEP 3 / PROJECTION',title:'Project the solid into the image.',body:'Imagine looking at each candidate through the camera. Its four solid rails predict which parts of the picture the gate would cover.',detail:'Moving and rotating the gate changes that projection, just as it does in the illustration above.',takeaway:'Compare the predicted view with the observed view.'},
  {kicker:'STEP 4 / COMPARISON',title:'Move the pose to improve agreement.',body:'Adjust the position and angle so the rails line up with target pixels and visible edges, while leaving clear background outside.',detail:'Treat missing evidence carefully. Cropping the picture does not create a new gate edge.',takeaway:'A closer visual match is useful, but is not proof of the exact pose.'},
  {kicker:'STEP 5 / ALTERNATIVES',title:'Keep the plausible explanations.',body:'Compare candidate gates together and keep alternatives when the picture cannot settle on one position or orientation.',detail:'Overlapping shapes or missing rails can leave several interpretations of the same scene.',takeaway:'Pass uncertainty forward instead of hiding it.'}
];
const baseRing = `<path d="M100 54H278V93H139V207H278V246H100Z" fill="#d8e3d2"/>`;
const fullOutline = `<path d="${ringPath(100,54,178,39)}" fill="none" stroke="#aa762c" stroke-width="2.2" stroke-dasharray="6 4"/>`;
function stepGraphic(index) {
  let v = `<title id="step-visual-title">${steps[index].title} Illustrative diagram.</title><rect x="17" y="17" width="406" height="260" rx="5" fill="#f1f4eb"/><path d="M17 147H423M219 17V277" stroke="#dbe2d5" stroke-dasharray="3 6"/>`;
  if(index===0) v+=`<path d="M100 54H278V93H139V207H278V246H100Z" fill="#286953"/><rect x="88" y="42" width="202" height="216" rx="4" stroke="#aa762c" stroke-dasharray="5 5" fill="none"/><text x="304" y="87">one region</text><path d="M292 78H300" stroke="#aa762c"/><text x="164" y="164">opening</text>`;
  if(index===1) v+=baseRing+`<g fill="none" stroke-width="2"><path d="${ringPath(89,47,185,41)}" stroke="#3e8165"/><path d="M110 39L292 62L276 256L98 229Z M144 86L248 100L240 208L139 194Z" stroke="#bd8a3d"/><path d="M122 59L277 36L296 226L135 249Z M155 96L243 83L256 196L163 210Z" stroke="#8d9b88"/></g><text x="305" y="99">pose A</text><text x="306" y="129">pose B</text><text x="306" y="159">pose C</text>`;
  if(index===2) v+=`<path d="M118 40L292 68L282 243L112 221Z M156 87L250 102L244 200L152 187Z" fill="#9ec5a9" fill-rule="evenodd"/><path d="M108 48L282 76L272 251L102 229Z M146 95L240 110L234 208L142 195Z" fill="#34795d" fill-rule="evenodd"/><path d="M108 48L118 40L292 68L282 76Z M282 76L292 68L282 243L272 251Z" fill="#639b7d"/><text x="306" y="154">project</text><text x="306" y="174">four rails</text>`;
  if(index===3){v+=baseRing+`<path d="${ringPath(105,49,181,39)}" fill="none" stroke="#236d51" stroke-width="2"/>`;[[115,65],[117,133],[118,220],[184,226],[205,75]].forEach(p=>v+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="#20684d"/>`);[[74,91],[161,156],[307,223],[318,112]].forEach(p=>v+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="#93a38c"/>`);[[100,110],[167,93],[206,207],[248,246]].forEach(p=>v+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="#b88029"/>`);v+=`<text x="318" y="60" fill="#246b52">inside</text><text x="318" y="80">outside</text><text x="318" y="100" fill="#a87327">interface</text>`;}
  if(index===4) v+=baseRing+fullOutline+`<path d="M112 46L286 61L280 251L103 235Z M150 88L246 98L241 209L144 201Z" fill="none" stroke="#407e62" stroke-width="2"/><rect x="304" y="111" width="97" height="27" rx="3" fill="#e0ead8"/><text x="315" y="129">mode A</text><rect x="304" y="149" width="97" height="27" rx="3" fill="#f0e4cc"/><text x="315" y="167">mode B</text><text x="301" y="202">both retained</text>`;
  document.querySelector('#step-visual').innerHTML=v;
}
let currentStep=0;
const tabs=[...document.querySelectorAll('[data-step]')];
function showStep(n,focus=false){
  currentStep=Math.max(0,Math.min(steps.length-1,n));const step=steps[currentStep];
  ['kicker','title','body','detail','takeaway'].forEach(key=>document.querySelector('#step-'+key).textContent=step[key]);
  tabs.forEach((tab,k)=>{tab.setAttribute('aria-selected',String(k===currentStep));tab.tabIndex=k===currentStep?0:-1;});
  document.querySelector('#step-panel').setAttribute('aria-labelledby','step-tab-'+currentStep);
  document.querySelector('#step-count').textContent=(currentStep+1)+' / '+steps.length;
  document.querySelector('#previous-step').disabled=currentStep===0;
  document.querySelector('#next-step').disabled=currentStep===steps.length-1;
  stepGraphic(currentStep);if(focus)tabs[currentStep].focus();
}
tabs.forEach((tab,i)=>{tab.addEventListener('click',()=>showStep(i));tab.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();showStep(next,true);}});});
document.querySelector('#previous-step').addEventListener('click',()=>showStep(currentStep-1));
document.querySelector('#next-step').addEventListener('click',()=>showStep(currentStep+1));
showStep(0);
// Print includes every walkthrough step, independent of the selected tab.
const printWalkthrough=document.createElement('div');
printWalkthrough.className='print-walkthrough';
for(const [i,step] of steps.entries()){
  const article=document.createElement('article');
  const heading=document.createElement('h3');heading.textContent=`${i+1}. ${step.title}`;
  const body=document.createElement('p');body.textContent=step.body+' '+step.detail;
  article.append(heading,body);printWalkthrough.append(article);
}
document.querySelector('.stepper').after(printWalkthrough);

// Select the section at the reading edge, even when a section exceeds the viewport.
const sectionLinks = [...document.querySelectorAll('.sidebar nav a')];
const readingSections = sectionLinks.map(link => document.querySelector(link.hash));
let navFrame = 0;
function updateReadingSection() {
  navFrame = 0;
  const topbar = document.querySelector('.topbar');
  const sidebar = document.querySelector('.sidebar');
  const horizontalNav = window.matchMedia('(max-width: 960px)').matches;
  const readingEdge = topbar.offsetHeight + (horizontalNav ? sidebar.offsetHeight : 0) + 45;
  let current = 0;
  readingSections.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= readingEdge) current = index;
  });
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 3) current = readingSections.length - 1;
  sectionLinks.forEach((link, index) => {
    link.classList.toggle('active', index === current);
    if (index === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
function queueNavigationUpdate() {
  if (!navFrame) navFrame = requestAnimationFrame(updateReadingSection);
}
window.addEventListener('scroll', queueNavigationUpdate, {passive: true});
window.addEventListener('resize', queueNavigationUpdate);
window.addEventListener('load', queueNavigationUpdate);
updateReadingSection();

// Separate tab groups: saved extraction images and conceptual fitting steps.
const exampleFrames = ['23503', '23495', '23216', '23586', '23279'];
const exampleStages = [
  {key:'source', label:'Source', description:'The camera captures the gates and the scene around them.', alt:'Source camera view of race gates in the simulator'},
  {key:'mask', label:'Target mask', description:'Selected target pixels separate likely gate material from the surrounding scene.', alt:'Camera view with retained target pixels highlighted'},
  {key:'boundary', label:'Boundary evidence', description:'The edges of each visible region provide clues about the gate’s shape.', alt:'A collection of boundary evidence views covering all detected regions'},
  {key:'lines', label:'Fitted lines', description:'Simple fitted shapes organize those clues into possible rail directions.', alt:'Fitted ellipses and lines over the full camera view'},
  {key:'candidates', label:'Gate candidates', description:'Known rail geometry helps compare candidate explanations for the target pixels.', alt:'Full-frame gate candidate decisions, with orange fallback and green validated-upgrade lines'}
];
let currentExample = '23503';
let currentExampleStage = 0;
const exampleTabs = [...document.querySelectorAll('[data-example-stage]')];
function showExample(stageIndex = currentExampleStage, focus = false) {
  currentExampleStage = stageIndex;
  const stage = exampleStages[stageIndex];
  const frameNumber = exampleFrames.indexOf(currentExample) + 1;
  const src = `assets/pipeline/frame-${currentExample}-${stage.key}.png`;
  const image = document.querySelector('#example-image');
  image.src = src;
  image.alt = `${stage.alt}, frame ${frameNumber} (capture ${currentExample}).`;
  document.querySelector('#example-image-link').href = src;
  document.querySelector('#example-image-link').setAttribute('aria-label', `Open ${stage.label.toLowerCase()} for frame ${frameNumber} at full size`);
  document.querySelector('#example-fullsize').href = src;
  document.querySelector('#example-stage-label').textContent = `${stageIndex + 1} / 5 · ${stage.label}`;
  document.querySelector('#example-caption').textContent = `Frame ${frameNumber} · ${stage.label}`;
  document.querySelector('#example-lesson').textContent = stage.description;
  document.querySelector('#example-panel').setAttribute('aria-labelledby', `example-tab-${stageIndex}`);
  document.querySelectorAll('[data-example]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.example === currentExample)));
  exampleTabs.forEach((tab, index) => {
    tab.setAttribute('aria-selected', String(index === stageIndex));
    tab.tabIndex = index === stageIndex ? 0 : -1;
  });
  if (focus) exampleTabs[stageIndex].focus();
}
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  currentExample = button.dataset.example;
  showExample();
}));
exampleTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => showExample(index));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % exampleTabs.length;
    if (event.key === 'ArrowLeft') next = (index + exampleTabs.length - 1) % exampleTabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = exampleTabs.length - 1;
    if (next !== undefined) { event.preventDefault(); showExample(next, true); }
  });
});
showExample();

// Schematic multiview sequence. Deliberately separate from recorded lab assets.
document.querySelectorAll('svg[data-temporal]').forEach(svg => {
  const stage = svg.dataset.temporal;
  const cameraA = '<path d="M42 126H63V146H42Z M63 131L73 126V146L63 141Z" fill="#286953"/>';
  const cameraB = '<path d="M158 126H179V146H158Z M179 131L189 126V146L179 141Z" fill="#aa762c"/>';
  let graphic = '<rect x="8" y="9" width="214" height="152" rx="3" fill="#f0f3e9"/>' +
    '<path d="M141 25H185V69H141Z M151 35V59H175V35Z" fill="#34795d" fill-rule="evenodd"/>' +
    '<path d="M62 128L163 47" stroke="#34795d" stroke-width="1.8"/>' + cameraA;
  if (stage !== 'identify') graphic += cameraB + '<path d="M82 145H144M137 140L144 145L137 150" fill="none" stroke="#9eac98"/>';
  if (stage === 'compare' || stage === 'check') graphic += '<path d="M168 126L163 47" stroke="#aa762c" stroke-width="1.8"/><circle cx="163" cy="47" r="4" fill="#f0b865"/>';
  if (stage === 'check') graphic += '<path d="M168 126L104 34" stroke="#9eac98" stroke-dasharray="4 4"/><path d="M104 70L114 80M114 70L104 80" stroke="#aa762c" stroke-width="2"/>';
  svg.innerHTML = graphic;
});
