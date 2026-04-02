const agentGridEl = document.getElementById("agentGrid");
const brainCanvas = document.getElementById("brain");
const bctx = brainCanvas.getContext("2d");

const generationEl = document.getElementById("generation");
const aliveAgentsEl = document.getElementById("aliveAgents");
const currentBestEl = document.getElementById("currentBest");
const globalBestEl = document.getElementById("globalBest");
const avgFitnessEl = document.getElementById("avgFitness");
const hardResetsEl = document.getElementById("hardResets");
const bestGenEl = document.getElementById("bestGen");
const speedEl = document.getElementById("speed");
const hiddenTargetEl = document.getElementById("hiddenTarget");
const avgHiddenEl = document.getElementById("avgHidden");
const avgLayersEl = document.getElementById("avgLayers");
const statusEl = document.getElementById("status");

const GRID = 20;
const POP_SIZE = 10;
const GOAL_SCORE = 100;
const INPUTS = 14;
const OUTPUTS = 3;

const DEFAULT_HIDDEN_LAYERS = 2;
const DEFAULT_HIDDEN_WIDTH = 12;
const MIN_HIDDEN_LAYERS = 1;
const MAX_HIDDEN_LAYERS = 6;
const MIN_NEURONS = 4;
const MAX_NEURONS = 30;

const ELITE_COUNT = 3;
const AGENT_CANVAS_SIZE = 220;

const CHECKPOINT_KEY = "snakeRLCheckpoint";
const CHECKPOINT_VERSION = 2;
const CHECKPOINT_SAVE_MS = 1500;

const TOPOLOGY_MUTATION_RATE = 0.22;

let agents = [];
let generation = 1;
let globalBestScore = 0;
let globalBestFitness = Number.NEGATIVE_INFINITY;
let globalBestGenome = null;
let bestFromGeneration = 1;
let hardResets = 0;
let paused = false;
let solved = false;
let stepsPerFrame = 6;
let statusText = "Training in progress";
let statusAlert = false;
let lastGenerationAverageFitness = 0;
let displayAgent = null;
let agentViews = new Map();
let lastCheckpointSaveAt = 0;

let preferredHiddenLayers = DEFAULT_HIDDEN_LAYERS;
let preferredHiddenWidth = DEFAULT_HIDDEN_WIDTH;

const DIRECTIONS = [
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
  { x: 0, y: -1 }
];

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function randInt(min, max) {
  return Math.floor(rand(min, max));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalRandom() {
  let u = 0;
  let v = 0;
  while (u === 0) {
    u = Math.random();
  }
  while (v === 0) {
    v = Math.random();
  }
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function saveBest(value) {
  localStorage.setItem("snakeRLBest", String(value));
}

function loadBest() {
  const saved = localStorage.getItem("snakeRLBest");
  return saved ? Number(saved) : 0;
}

function randomHiddenSize() {
  return clamp(randInt(preferredHiddenWidth - 3, preferredHiddenWidth + 4), MIN_NEURONS, MAX_NEURONS);
}

function randomHiddenSizes(layerCount = preferredHiddenLayers) {
  return Array.from({ length: clamp(layerCount, MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS) }, () => randomHiddenSize());
}

function createMatrix(rows, cols, scale = 1) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rand(-1, 1) * scale)
  );
}

function createVector(length, scale = 0.4) {
  return Array.from({ length }, () => rand(-1, 1) * scale);
}

function createGenomeFromHiddenSizes(hiddenSizes) {
  const validHidden = hiddenSizes.map((size) => clamp(size, MIN_NEURONS, MAX_NEURONS));
  const allLayerSizes = [INPUTS, ...validHidden, OUTPUTS];

  const weights = [];
  const biases = [];
  for (let i = 0; i < allLayerSizes.length - 1; i += 1) {
    weights.push(createMatrix(allLayerSizes[i + 1], allLayerSizes[i]));
    biases.push(createVector(allLayerSizes[i + 1]));
  }

  return {
    hiddenSizes: validHidden,
    weights,
    biases
  };
}

function createRandomGenome() {
  return createGenomeFromHiddenSizes(randomHiddenSizes());
}

function cloneGenome(genome) {
  return {
    hiddenSizes: genome.hiddenSizes.slice(),
    weights: genome.weights.map((matrix) => matrix.map((row) => row.slice())),
    biases: genome.biases.map((vector) => vector.slice())
  };
}

function isValidGenome(genome) {
  if (!genome || !Array.isArray(genome.hiddenSizes) || !Array.isArray(genome.weights) || !Array.isArray(genome.biases)) {
    return false;
  }

  const hiddenCount = genome.hiddenSizes.length;
  if (hiddenCount < MIN_HIDDEN_LAYERS || hiddenCount > MAX_HIDDEN_LAYERS) {
    return false;
  }

  if (genome.weights.length !== hiddenCount + 1 || genome.biases.length !== hiddenCount + 1) {
    return false;
  }

  for (let i = 0; i < hiddenCount; i += 1) {
    const s = genome.hiddenSizes[i];
    if (!Number.isFinite(s) || s < MIN_NEURONS || s > MAX_NEURONS) {
      return false;
    }
  }

  const layerSizes = [INPUTS, ...genome.hiddenSizes, OUTPUTS];
  for (let i = 0; i < layerSizes.length - 1; i += 1) {
    const rows = layerSizes[i + 1];
    const cols = layerSizes[i];
    const matrix = genome.weights[i];
    const vector = genome.biases[i];

    if (!Array.isArray(matrix) || matrix.length !== rows) {
      return false;
    }
    if (!matrix.every((row) => Array.isArray(row) && row.length === cols)) {
      return false;
    }
    if (!Array.isArray(vector) || vector.length !== rows) {
      return false;
    }
  }

  return true;
}

function remapGenomeToHiddenSizes(oldGenome, newHiddenSizes) {
  const genome = createGenomeFromHiddenSizes(newHiddenSizes);
  const maxLayers = Math.min(oldGenome.weights.length, genome.weights.length);

  for (let layer = 0; layer < maxLayers; layer += 1) {
    const oldW = oldGenome.weights[layer];
    const newW = genome.weights[layer];
    const rows = Math.min(oldW.length, newW.length);

    for (let r = 0; r < rows; r += 1) {
      const cols = Math.min(oldW[r].length, newW[r].length);
      for (let c = 0; c < cols; c += 1) {
        if (Math.random() < 0.75) {
          newW[r][c] = oldW[r][c];
        }
      }
    }

    const oldB = oldGenome.biases[layer];
    const newB = genome.biases[layer];
    const bLen = Math.min(oldB.length, newB.length);
    for (let i = 0; i < bLen; i += 1) {
      if (Math.random() < 0.75) {
        newB[i] = oldB[i];
      }
    }
  }

  return genome;
}

function crossover(a, b) {
  const useA = Math.random() < 0.5;
  const base = useA ? a : b;
  const other = useA ? b : a;

  const hiddenCount = clamp(
    Math.random() < 0.5 ? base.hiddenSizes.length : other.hiddenSizes.length,
    MIN_HIDDEN_LAYERS,
    MAX_HIDDEN_LAYERS
  );

  const childHiddenSizes = [];
  for (let i = 0; i < hiddenCount; i += 1) {
    const fromBase = i < base.hiddenSizes.length ? base.hiddenSizes[i] : randomHiddenSize();
    const fromOther = i < other.hiddenSizes.length ? other.hiddenSizes[i] : randomHiddenSize();
    const val = Math.random() < 0.5 ? fromBase : fromOther;
    childHiddenSizes.push(clamp(val, MIN_NEURONS, MAX_NEURONS));
  }

  const child = createGenomeFromHiddenSizes(childHiddenSizes);

  const maxLayers = Math.min(child.weights.length, base.weights.length, other.weights.length);
  for (let layer = 0; layer < maxLayers; layer += 1) {
    const matrix = child.weights[layer];
    for (let r = 0; r < matrix.length; r += 1) {
      for (let c = 0; c < matrix[r].length; c += 1) {
        const av = r < base.weights[layer].length && c < base.weights[layer][r].length ? base.weights[layer][r][c] : rand(-1, 1);
        const bv = r < other.weights[layer].length && c < other.weights[layer][r].length ? other.weights[layer][r][c] : rand(-1, 1);
        matrix[r][c] = Math.random() < 0.5 ? av : bv;
      }
    }

    const bvec = child.biases[layer];
    for (let i = 0; i < bvec.length; i += 1) {
      const av = i < base.biases[layer].length ? base.biases[layer][i] : rand(-0.4, 0.4);
      const bv = i < other.biases[layer].length ? other.biases[layer][i] : rand(-0.4, 0.4);
      bvec[i] = Math.random() < 0.5 ? av : bv;
    }
  }

  return child;
}

function mutateWeightsAndBiases(genome, rate, strength) {
  for (let layer = 0; layer < genome.weights.length; layer += 1) {
    for (let r = 0; r < genome.weights[layer].length; r += 1) {
      for (let c = 0; c < genome.weights[layer][r].length; c += 1) {
        if (Math.random() < rate) {
          genome.weights[layer][r][c] += normalRandom() * strength;
        }
      }
    }

    for (let i = 0; i < genome.biases[layer].length; i += 1) {
      if (Math.random() < rate) {
        genome.biases[layer][i] += normalRandom() * strength * 0.5;
      }
    }
  }
}

function mutateTopology(genome) {
  let copy = cloneGenome(genome);

  const maybeAddLayer = Math.random() < TOPOLOGY_MUTATION_RATE && copy.hiddenSizes.length < MAX_HIDDEN_LAYERS;
  const maybeRemoveLayer = Math.random() < TOPOLOGY_MUTATION_RATE && copy.hiddenSizes.length > MIN_HIDDEN_LAYERS;

  if (maybeAddLayer) {
    const idx = randInt(0, copy.hiddenSizes.length + 1);
    const newHidden = copy.hiddenSizes.slice();
    newHidden.splice(idx, 0, randomHiddenSize());
    copy = remapGenomeToHiddenSizes(copy, newHidden);
  }

  if (maybeRemoveLayer && copy.hiddenSizes.length > MIN_HIDDEN_LAYERS) {
    const idx = randInt(0, copy.hiddenSizes.length);
    const newHidden = copy.hiddenSizes.slice();
    newHidden.splice(idx, 1);
    copy = remapGenomeToHiddenSizes(copy, newHidden);
  }

  if (Math.random() < TOPOLOGY_MUTATION_RATE * 1.5) {
    const layer = randInt(0, copy.hiddenSizes.length);
    const delta = Math.random() < 0.5 ? -1 : 1;
    const newHidden = copy.hiddenSizes.slice();
    newHidden[layer] = clamp(newHidden[layer] + delta, MIN_NEURONS, MAX_NEURONS);
    copy = remapGenomeToHiddenSizes(copy, newHidden);
  }

  const biasTowardPreferred = Math.random() < 0.42;
  if (biasTowardPreferred) {
    const target = [];
    for (let i = 0; i < preferredHiddenLayers; i += 1) {
      const base = i < copy.hiddenSizes.length ? copy.hiddenSizes[i] : preferredHiddenWidth;
      const nudged = Math.round(base * 0.7 + preferredHiddenWidth * 0.3);
      target.push(clamp(nudged, MIN_NEURONS, MAX_NEURONS));
    }
    copy = remapGenomeToHiddenSizes(copy, target);
  }

  return copy;
}

function mutate(genome, rate = 0.16, strength = 0.22) {
  let copy = cloneGenome(genome);
  mutateWeightsAndBiases(copy, rate, strength);
  copy = mutateTopology(copy);
  return copy;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
  return Math.tanh(x);
}

function forward(genome, inputValues) {
  const activations = [inputValues.slice()];
  let current = inputValues.slice();

  for (let layer = 0; layer < genome.weights.length; layer += 1) {
    const matrix = genome.weights[layer];
    const bias = genome.biases[layer];
    const next = Array(matrix.length).fill(0);

    for (let r = 0; r < matrix.length; r += 1) {
      let sum = bias[r];
      for (let c = 0; c < matrix[r].length; c += 1) {
        sum += matrix[r][c] * current[c];
      }
      next[r] = layer === genome.weights.length - 1 ? sum : tanh(sum);
    }

    current = next;
    activations.push(next.slice());
  }

  const hiddenLayers = activations.slice(1, activations.length - 1);
  const outputs = activations[activations.length - 1];
  return { outputs, hiddenLayers, layerActivations: activations };
}

function scoreToFitness(agent) {
  const distancePenalty = manhattan(agent.body[0], agent.food) * 0.7;
  const topologyBonus = agent.genome.hiddenSizes.reduce((sum, v) => sum + v, 0) * 0.3;
  return agent.score * 1800 + agent.steps * 1.2 - distancePenalty + agent.maxLength * 15 + topologyBonus;
}

function manhattan(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function spawnFood(body) {
  while (true) {
    const point = { x: randInt(0, GRID), y: randInt(0, GRID) };
    const occupied = body.some((cell) => cell.x === point.x && cell.y === point.y);
    if (!occupied) {
      return point;
    }
  }
}

function createAgent(genome, id) {
  const startX = randInt(4, GRID - 4);
  const startY = randInt(4, GRID - 4);
  const body = [
    { x: startX, y: startY },
    { x: startX - 1, y: startY },
    { x: startX - 2, y: startY }
  ];

  return {
    id,
    genome,
    body,
    dirIndex: 0,
    food: spawnFood(body),
    alive: true,
    score: 0,
    steps: 0,
    hunger: 0,
    maxLength: body.length,
    fitness: 0,
    lastInputs: Array(INPUTS).fill(0),
    lastOutputs: Array(OUTPUTS).fill(0),
    lastForward: null
  };
}

function createAgentCard(id) {
  const card = document.createElement("article");
  card.className = "agent-card";

  const head = document.createElement("div");
  head.className = "agent-head";

  const idEl = document.createElement("span");
  idEl.className = "agent-id";
  idEl.textContent = `A${id}`;

  const stateEl = document.createElement("span");
  stateEl.className = "agent-state alive";
  stateEl.textContent = "ALIVE";

  head.appendChild(idEl);
  head.appendChild(stateEl);

  const scoreEl = document.createElement("div");
  scoreEl.className = "agent-score";
  scoreEl.textContent = "S:0 | T:0 | L:2";

  const board = document.createElement("canvas");
  board.className = "agent-canvas";
  board.width = AGENT_CANVAS_SIZE;
  board.height = AGENT_CANVAS_SIZE;
  board.setAttribute("aria-label", `Agent ${id} board`);

  card.appendChild(head);
  card.appendChild(scoreEl);
  card.appendChild(board);
  agentGridEl.appendChild(card);

  return {
    card,
    stateEl,
    scoreEl,
    board,
    ctx: board.getContext("2d")
  };
}

function ensureAgentViews() {
  if (agentViews.size === POP_SIZE) {
    return;
  }

  agentViews = new Map();
  agentGridEl.innerHTML = "";
  for (let id = 1; id <= POP_SIZE; id += 1) {
    agentViews.set(id, createAgentCard(id));
  }
}

function getDirVector(dirIndex) {
  return DIRECTIONS[(dirIndex + 4) % 4];
}

function turn(dirIndex, action) {
  if (action === 1) {
    return (dirIndex + 3) % 4;
  }
  if (action === 2) {
    return (dirIndex + 1) % 4;
  }
  return dirIndex;
}

function isCollision(agent, x, y) {
  if (x < 0 || x >= GRID || y < 0 || y >= GRID) {
    return true;
  }
  return agent.body.some((part) => part.x === x && part.y === y);
}

function rayDistance(agent, directionVector) {
  let x = agent.body[0].x;
  let y = agent.body[0].y;
  let distance = 0;

  while (true) {
    x += directionVector.x;
    y += directionVector.y;
    distance += 1;
    if (x < 0 || x >= GRID || y < 0 || y >= GRID) {
      break;
    }
    if (agent.body.some((part) => part.x === x && part.y === y)) {
      break;
    }
  }

  return distance / GRID;
}

function getInputs(agent) {
  const head = agent.body[0];
  const forwardVec = getDirVector(agent.dirIndex);
  const leftVec = getDirVector(agent.dirIndex - 1);
  const rightVec = getDirVector(agent.dirIndex + 1);

  const aheadCollision = isCollision(agent, head.x + forwardVec.x, head.y + forwardVec.y) ? 1 : 0;
  const leftCollision = isCollision(agent, head.x + leftVec.x, head.y + leftVec.y) ? 1 : 0;
  const rightCollision = isCollision(agent, head.x + rightVec.x, head.y + rightVec.y) ? 1 : 0;

  const foodDx = agent.food.x - head.x;
  const foodDy = agent.food.y - head.y;
  const foodAhead = forwardVec.x * foodDx + forwardVec.y * foodDy > 0 ? 1 : 0;
  const foodLeft = leftVec.x * foodDx + leftVec.y * foodDy > 0 ? 1 : 0;
  const foodRight = rightVec.x * foodDx + rightVec.y * foodDy > 0 ? 1 : 0;

  const heading = [0, 0, 0, 0];
  heading[agent.dirIndex] = 1;

  return [
    aheadCollision,
    leftCollision,
    rightCollision,
    foodAhead,
    foodLeft,
    foodRight,
    clamp(foodDx / GRID, -1, 1),
    clamp(foodDy / GRID, -1, 1),
    heading[0],
    heading[1],
    heading[2],
    heading[3],
    rayDistance(agent, forwardVec),
    clamp(agent.body.length / 30, 0, 1)
  ];
}

function chooseAction(outputs) {
  let bestIdx = 0;
  let bestVal = outputs[0];
  for (let i = 1; i < outputs.length; i += 1) {
    if (outputs[i] > bestVal) {
      bestVal = outputs[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

function killAgent(agent) {
  if (!agent.alive) {
    return;
  }
  agent.alive = false;
  agent.fitness = scoreToFitness(agent);
}

function evaluateOneStep(agent) {
  if (!agent.alive) {
    return;
  }

  const inputs = getInputs(agent);
  const result = forward(agent.genome, inputs);
  const action = chooseAction(result.outputs);
  agent.dirIndex = turn(agent.dirIndex, action);
  const direction = getDirVector(agent.dirIndex);

  const nextHead = {
    x: agent.body[0].x + direction.x,
    y: agent.body[0].y + direction.y
  };

  const hitsWall = nextHead.x < 0 || nextHead.x >= GRID || nextHead.y < 0 || nextHead.y >= GRID;
  const hitsBody = agent.body.some((part) => part.x === nextHead.x && part.y === nextHead.y);
  if (hitsWall || hitsBody) {
    killAgent(agent);
    return;
  }

  agent.body.unshift(nextHead);
  const ateFood = nextHead.x === agent.food.x && nextHead.y === agent.food.y;

  if (ateFood) {
    agent.score += 1;
    agent.hunger = 0;
    agent.food = spawnFood(agent.body);
    if (agent.body.length > agent.maxLength) {
      agent.maxLength = agent.body.length;
    }
  } else {
    agent.body.pop();
    agent.hunger += 1;
  }

  agent.steps += 1;
  const starvationLimit = 75 + agent.score * 25;
  if (agent.hunger > starvationLimit) {
    killAgent(agent);
  }

  agent.lastInputs = inputs;
  agent.lastOutputs = result.outputs;
  agent.lastForward = result;
}

function initializePopulation(genomes) {
  ensureAgentViews();
  agents = genomes.map((genome, i) => createAgent(genome, i + 1));
}

function fitGenomeToPreferred(genome) {
  const hiddenSizes = [];
  for (let i = 0; i < preferredHiddenLayers; i += 1) {
    const old = i < genome.hiddenSizes.length ? genome.hiddenSizes[i] : preferredHiddenWidth;
    const blended = Math.round(old * 0.65 + preferredHiddenWidth * 0.35);
    hiddenSizes.push(clamp(blended, MIN_NEURONS, MAX_NEURONS));
  }
  return remapGenomeToHiddenSizes(genome, hiddenSizes);
}

function applyPreferredTopologyNow() {
  if (globalBestGenome) {
    globalBestGenome = fitGenomeToPreferred(globalBestGenome);
  }

  agents = agents.map((agent) => ({
    ...agent,
    genome: fitGenomeToPreferred(agent.genome)
  }));

  saveCheckpoint(true);
}

function clearCheckpoint() {
  localStorage.removeItem(CHECKPOINT_KEY);
}

function saveCheckpoint(force = false) {
  const now = performance.now();
  if (!force && now - lastCheckpointSaveAt < CHECKPOINT_SAVE_MS) {
    return;
  }

  if (!agents.length) {
    return;
  }

  const payload = {
    version: CHECKPOINT_VERSION,
    timestamp: Date.now(),
    generation,
    globalBestScore,
    globalBestFitness,
    bestFromGeneration,
    hardResets,
    solved,
    stepsPerFrame,
    preferredHiddenLayers,
    preferredHiddenWidth,
    globalBestGenome: globalBestGenome ? cloneGenome(globalBestGenome) : null,
    populationGenomes: agents.map((agent) => cloneGenome(agent.genome))
  };

  try {
    localStorage.setItem(CHECKPOINT_KEY, JSON.stringify(payload));
    lastCheckpointSaveAt = now;
  } catch (error) {
    setStatus("Autosave failed: storage is full.", true);
  }
}

function loadCheckpoint() {
  try {
    const raw = localStorage.getItem(CHECKPOINT_KEY);
    if (!raw) {
      return null;
    }

    const checkpoint = JSON.parse(raw);
    if (!checkpoint || checkpoint.version !== CHECKPOINT_VERSION) {
      return null;
    }

    if (!Array.isArray(checkpoint.populationGenomes) || checkpoint.populationGenomes.length !== POP_SIZE) {
      return null;
    }

    const allValid = checkpoint.populationGenomes.every((genome) => isValidGenome(genome));
    if (!allValid) {
      return null;
    }

    if (checkpoint.globalBestGenome && !isValidGenome(checkpoint.globalBestGenome)) {
      return null;
    }

    return checkpoint;
  } catch (error) {
    return null;
  }
}

function restoreFromCheckpoint(checkpoint) {
  generation = Number(checkpoint.generation) || 1;
  globalBestScore = Number(checkpoint.globalBestScore) || 0;
  globalBestFitness = Number(checkpoint.globalBestFitness) || Number.NEGATIVE_INFINITY;
  bestFromGeneration = Number(checkpoint.bestFromGeneration) || generation;
  hardResets = Number(checkpoint.hardResets) || 0;
  solved = Boolean(checkpoint.solved);
  stepsPerFrame = clamp(Number(checkpoint.stepsPerFrame) || 6, 1, 30);

  preferredHiddenLayers = clamp(Number(checkpoint.preferredHiddenLayers) || DEFAULT_HIDDEN_LAYERS, MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS);
  preferredHiddenWidth = clamp(Number(checkpoint.preferredHiddenWidth) || DEFAULT_HIDDEN_WIDTH, MIN_NEURONS, MAX_NEURONS);

  globalBestGenome = checkpoint.globalBestGenome ? cloneGenome(checkpoint.globalBestGenome) : createRandomGenome();

  initializePopulation(checkpoint.populationGenomes.map((genome) => cloneGenome(genome)));

  if (globalBestScore > 0) {
    saveBest(globalBestScore);
  }

  setStatus("Checkpoint restored. Training resumed.", true);
}

function weightedPick(pool) {
  const total = pool.reduce((sum, item) => sum + item.weight, 0);
  const r = rand(0, total);
  let accum = 0;
  for (let i = 0; i < pool.length; i += 1) {
    accum += pool[i].weight;
    if (r <= accum) {
      return pool[i].genome;
    }
  }
  return pool[0].genome;
}

function createNextGenerationFromTop(snakes) {
  const sorted = snakes.slice().sort((a, b) => b.fitness - a.fitness);
  const elites = sorted.slice(0, ELITE_COUNT);
  const parentPool = elites.map((agent, idx) => ({
    genome: cloneGenome(agent.genome),
    weight: ELITE_COUNT - idx + 1
  }));

  const nextGenomes = [];
  nextGenomes.push(cloneGenome(elites[0].genome));

  for (let i = 1; i < POP_SIZE; i += 1) {
    const a = weightedPick(parentPool);
    const b = weightedPick(parentPool);
    let child = crossover(a, b);

    const intensity = i < Math.ceil(POP_SIZE * 0.4) ? 0.09 : 0.2;
    const mutationRate = i < Math.ceil(POP_SIZE * 0.4) ? 0.12 : 0.22;
    child = mutate(child, mutationRate, intensity);

    if (globalBestGenome && i % 7 === 0) {
      child = mutate(cloneGenome(globalBestGenome), 0.1, 0.11);
    }

    nextGenomes.push(child);
  }

  return { nextGenomes, generationBest: elites[0] };
}

function setStatus(message, isAlert = false) {
  statusText = message;
  statusAlert = isAlert;
  statusEl.textContent = message;
  statusEl.classList.toggle("alert", isAlert);
}

function hardResetWithBest(bestAgent) {
  globalBestGenome = cloneGenome(bestAgent.genome);
  globalBestScore = bestAgent.score;
  globalBestFitness = bestAgent.fitness;
  bestFromGeneration = generation;
  hardResets += 1;
  saveBest(globalBestScore);

  const nextGenomes = [];
  for (let i = 0; i < POP_SIZE; i += 1) {
    const source = cloneGenome(globalBestGenome);
    nextGenomes.push(i === 0 ? source : mutate(source, 0.22, 0.2));
  }

  generation += 1;
  initializePopulation(nextGenomes);
  setStatus("New best found. Champion cloned to all agents.", true);
  saveCheckpoint(true);
}

function startFromScratch() {
  generation = 1;
  hardResets = 0;
  solved = false;
  globalBestFitness = Number.NEGATIVE_INFINITY;
  globalBestGenome = createRandomGenome();
  globalBestScore = loadBest();

  const genomes = [];
  for (let i = 0; i < POP_SIZE; i += 1) {
    const base = i === 0 ? cloneGenome(globalBestGenome) : mutate(globalBestGenome, 0.26, 0.35);
    genomes.push(base);
  }

  initializePopulation(genomes);
  setStatus("Training in progress", false);
  clearCheckpoint();
  saveCheckpoint(true);
}

function evolveIfNeeded() {
  const alive = agents.filter((a) => a.alive).length;
  if (alive > 0 || solved) {
    return;
  }

  agents.forEach((agent) => {
    if (!agent.fitness) {
      agent.fitness = scoreToFitness(agent);
    }
  });

  const avgFit = agents.reduce((sum, agent) => sum + agent.fitness, 0) / agents.length;
  lastGenerationAverageFitness = avgFit;

  const { nextGenomes, generationBest } = createNextGenerationFromTop(agents);
  if (generationBest.score > globalBestScore) {
    hardResetWithBest(generationBest);
    return;
  }

  if (generationBest.fitness > globalBestFitness) {
    globalBestFitness = generationBest.fitness;
    globalBestGenome = cloneGenome(generationBest.genome);
    bestFromGeneration = generation;
  }

  generation += 1;
  initializePopulation(nextGenomes);
  setStatus("Generation replaced from top performers.", false);
  saveCheckpoint(true);
}

function trainStep() {
  if (paused || solved) {
    return;
  }

  for (let tick = 0; tick < stepsPerFrame; tick += 1) {
    for (let i = 0; i < agents.length; i += 1) {
      evaluateOneStep(agents[i]);

      if (agents[i].score > globalBestScore) {
        agents[i].fitness = scoreToFitness(agents[i]);
        if (agents[i].score >= GOAL_SCORE) {
          globalBestScore = agents[i].score;
          globalBestGenome = cloneGenome(agents[i].genome);
          bestFromGeneration = generation;
          saveBest(globalBestScore);
          solved = true;
          setStatus("Goal reached: score 100 achieved by champion snake.", true);
          saveCheckpoint(true);
          return;
        }
        hardResetWithBest(agents[i]);
        return;
      }
    }

    evolveIfNeeded();
  }
}

function pickDisplayAgent() {
  const alive = agents.filter((agent) => agent.alive);
  if (alive.length === 0) {
    displayAgent = agents[0] || null;
    return;
  }

  alive.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return b.steps - a.steps;
  });
  displayAgent = alive[0];
}

function drawAgentBoard(agent, boardCtx, boardCanvas) {
  if (!agent) {
    return;
  }

  const tile = boardCanvas.width / GRID;

  boardCtx.fillStyle = "#0f1320";
  boardCtx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

  boardCtx.strokeStyle = "rgba(255,255,255,0.08)";
  boardCtx.lineWidth = 1;
  for (let i = 0; i <= GRID; i += 1) {
    const p = i * tile;
    boardCtx.beginPath();
    boardCtx.moveTo(p, 0);
    boardCtx.lineTo(p, boardCanvas.height);
    boardCtx.stroke();
    boardCtx.beginPath();
    boardCtx.moveTo(0, p);
    boardCtx.lineTo(boardCanvas.width, p);
    boardCtx.stroke();
  }

  boardCtx.fillStyle = "#ff7a45";
  boardCtx.beginPath();
  boardCtx.arc(
    agent.food.x * tile + tile / 2,
    agent.food.y * tile + tile / 2,
    tile * 0.34,
    0,
    Math.PI * 2
  );
  boardCtx.fill();

  for (let i = agent.body.length - 1; i >= 0; i -= 1) {
    const segment = agent.body[i];
    const alpha = i === 0 ? 1 : clamp(0.35 + (agent.body.length - i) * 0.015, 0.35, 0.8);
    boardCtx.fillStyle = i === 0 ? "#43d692" : `rgba(67, 214, 146, ${alpha})`;
    boardCtx.fillRect(segment.x * tile + 1, segment.y * tile + 1, tile - 2, tile - 2);
  }
}

function drawAllAgentBoards() {
  const ordered = agents.slice().sort((a, b) => {
    if (a.alive !== b.alive) {
      return a.alive ? -1 : 1;
    }
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    const af = a.fitness || scoreToFitness(a);
    const bf = b.fitness || scoreToFitness(b);
    if (bf !== af) {
      return bf - af;
    }
    if (b.steps !== a.steps) {
      return b.steps - a.steps;
    }
    return a.id - b.id;
  });

  const bestId = ordered.length ? ordered[0].id : -1;
  const midId = ordered.length ? ordered[Math.floor((ordered.length - 1) / 2)].id : -1;
  const worstId = ordered.length ? ordered[ordered.length - 1].id : -1;

  for (let i = 0; i < ordered.length; i += 1) {
    const agent = ordered[i];
    const view = agentViews.get(agent.id);
    if (!view) {
      continue;
    }

    agentGridEl.appendChild(view.card);
    drawAgentBoard(agent, view.ctx, view.board);

    if (paused || solved) {
      view.ctx.fillStyle = "rgba(0,0,0,0.4)";
      view.ctx.fillRect(0, 0, view.board.width, view.board.height);
    }

    view.stateEl.textContent = agent.alive ? "ALIVE" : "DEAD";
    view.stateEl.classList.toggle("alive", agent.alive);
    view.stateEl.classList.toggle("dead", !agent.alive);
    view.scoreEl.textContent = `S:${agent.score} | T:${agent.steps} | L:${agent.genome.hiddenSizes.length}`;

    view.card.classList.toggle("rank-best", agent.id === bestId);
    view.card.classList.toggle("rank-mid", agent.id === midId);
    view.card.classList.toggle("rank-worst", agent.id === worstId);
  }
}

function drawNetwork(agent) {
  const width = brainCanvas.width;
  const height = brainCanvas.height;

  bctx.clearRect(0, 0, width, height);
  bctx.fillStyle = "#0a0d17";
  bctx.fillRect(0, 0, width, height);

  if (!agent) {
    return;
  }

  const result = forward(agent.genome, agent.lastInputs || Array(INPUTS).fill(0));
  const layerSizes = [INPUTS, ...agent.genome.hiddenSizes, OUTPUTS];
  const xMargin = 28;
  const yMargin = 20;

  const layerPositions = layerSizes.map((size, layerIdx) => {
    const x = xMargin + (layerIdx * (width - xMargin * 2)) / (layerSizes.length - 1);
    return Array.from({ length: size }, (_, i) => ({
      x,
      y: size === 1
        ? height / 2
        : yMargin + (i * (height - yMargin * 2)) / (size - 1)
    }));
  });

  for (let layer = 0; layer < agent.genome.weights.length; layer += 1) {
    const src = layerPositions[layer];
    const dst = layerPositions[layer + 1];
    const weights = agent.genome.weights[layer];

    for (let r = 0; r < weights.length; r += 1) {
      for (let c = 0; c < weights[r].length; c += 1) {
        const w = weights[r][c];
        bctx.strokeStyle = w >= 0
          ? `rgba(67,214,146,${Math.min(0.65, Math.abs(w) * 0.23)})`
          : `rgba(255,122,69,${Math.min(0.65, Math.abs(w) * 0.23)})`;
        bctx.lineWidth = layer === agent.genome.weights.length - 1 ? 1.4 : 1;
        bctx.beginPath();
        bctx.moveTo(src[c].x, src[c].y);
        bctx.lineTo(dst[r].x, dst[r].y);
        bctx.stroke();
      }
    }
  }

  const activationLayers = result.layerActivations;
  for (let layer = 0; layer < layerPositions.length; layer += 1) {
    const positions = layerPositions[layer];
    for (let i = 0; i < positions.length; i += 1) {
      const value = layer === 0
        ? sigmoid((agent.lastInputs || [])[i] || 0)
        : sigmoid((activationLayers[layer] || [])[i] || 0);

      const isOutput = layer === layerPositions.length - 1;
      bctx.fillStyle = isOutput
        ? `rgba(255,122,69,${0.3 + value * 0.65})`
        : `rgba(126,181,255,${0.22 + value * 0.68})`;

      if (layer > 0 && layer < layerPositions.length - 1) {
        bctx.fillStyle = `rgba(67,214,146,${0.2 + value * 0.75})`;
      }

      bctx.beginPath();
      bctx.arc(positions[i].x, positions[i].y, isOutput ? 5.8 : 3.8, 0, Math.PI * 2);
      bctx.fill();
    }
  }

  bctx.fillStyle = "#8ca2d4";
  bctx.font = "10px Consolas";
  bctx.textAlign = "left";
  bctx.fillText(`Layers: ${agent.genome.hiddenSizes.length} hidden`, 8, 12);
  bctx.fillText(`Hidden sizes: ${agent.genome.hiddenSizes.join("-")}`, 8, 25);
}

function updateHud() {
  const alive = agents.filter((agent) => agent.alive);
  const currentBest = agents.reduce((best, agent) => Math.max(best, agent.score), 0);

  const avgHiddenNeurons = agents.length
    ? agents.reduce((sum, agent) => {
      const total = agent.genome.hiddenSizes.reduce((s, n) => s + n, 0);
      return sum + total / agent.genome.hiddenSizes.length;
    }, 0) / agents.length
    : preferredHiddenWidth;

  const avgHiddenLayers = agents.length
    ? agents.reduce((sum, agent) => sum + agent.genome.hiddenSizes.length, 0) / agents.length
    : preferredHiddenLayers;

  const avgFitness = alive.length
    ? alive.reduce((sum, agent) => sum + scoreToFitness(agent), 0) / alive.length
    : lastGenerationAverageFitness;

  generationEl.textContent = String(generation);
  aliveAgentsEl.textContent = String(alive.length);
  currentBestEl.textContent = String(currentBest);
  globalBestEl.textContent = String(globalBestScore);
  avgFitnessEl.textContent = avgFitness.toFixed(1);
  hardResetsEl.textContent = String(hardResets);
  bestGenEl.textContent = String(bestFromGeneration);
  speedEl.textContent = `${stepsPerFrame}x`;
  hiddenTargetEl.textContent = `${preferredHiddenLayers}x${preferredHiddenWidth}`;
  avgHiddenEl.textContent = avgHiddenNeurons.toFixed(1);
  avgLayersEl.textContent = avgHiddenLayers.toFixed(1);

  statusEl.textContent = statusText;
  statusEl.classList.toggle("alert", statusAlert);
}

function draw() {
  pickDisplayAgent();
  drawAllAgentBoards();
  drawNetwork(displayAgent);
  updateHud();
}

function loop() {
  trainStep();
  draw();
  saveCheckpoint(false);
  requestAnimationFrame(loop);
}

document.addEventListener("keydown", (event) => {
  const key = event.key;

  if (key === " ") {
    paused = !paused;
    if (paused) {
      setStatus("Training paused", true);
    } else if (!solved) {
      setStatus("Training in progress", false);
    }
  } else if (key === "[") {
    stepsPerFrame = clamp(stepsPerFrame - 1, 1, 30);
    saveCheckpoint(true);
  } else if (key === "]") {
    stepsPerFrame = clamp(stepsPerFrame + 1, 1, 30);
    saveCheckpoint(true);
  } else if (key.toLowerCase() === "h") {
    preferredHiddenWidth = clamp(preferredHiddenWidth + 1, MIN_NEURONS, MAX_NEURONS);
    applyPreferredTopologyNow();
    setStatus(`Hidden width target increased to ${preferredHiddenWidth}.`, false);
  } else if (key.toLowerCase() === "j") {
    preferredHiddenWidth = clamp(preferredHiddenWidth - 1, MIN_NEURONS, MAX_NEURONS);
    applyPreferredTopologyNow();
    setStatus(`Hidden width target decreased to ${preferredHiddenWidth}.`, false);
  } else if (key.toLowerCase() === "u") {
    preferredHiddenLayers = clamp(preferredHiddenLayers + 1, MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS);
    applyPreferredTopologyNow();
    setStatus(`Hidden layer depth target increased to ${preferredHiddenLayers}.`, false);
  } else if (key.toLowerCase() === "k") {
    preferredHiddenLayers = clamp(preferredHiddenLayers - 1, MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS);
    applyPreferredTopologyNow();
    setStatus(`Hidden layer depth target decreased to ${preferredHiddenLayers}.`, false);
  } else if (key.toLowerCase() === "r") {
    startFromScratch();
    setStatus("Manual hard reset. Training restarted from scratch.", true);
  }
});

const checkpoint = loadCheckpoint();
if (checkpoint) {
  restoreFromCheckpoint(checkpoint);
} else {
  startFromScratch();
}

loop();
