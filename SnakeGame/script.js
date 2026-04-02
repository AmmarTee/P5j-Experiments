const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const scoreEl = document.getElementById("score");
const bestEl = document.getElementById("best");
const statusEl = document.getElementById("status");

const gridSize = 20;
const tileCount = canvas.width / gridSize;
const tickMs = 110;

let snake;
let direction;
let queuedDirection;
let food;
let score;
let best;
let started;
let paused;
let gameOver;
let loopId;

function loadBest() {
  const saved = localStorage.getItem("snakeBest");
  return saved ? Number(saved) : 0;
}

function saveBest(value) {
  localStorage.setItem("snakeBest", String(value));
}

function resetGame() {
  snake = [
    { x: 10, y: 10 },
    { x: 9, y: 10 },
    { x: 8, y: 10 }
  ];
  direction = { x: 1, y: 0 };
  queuedDirection = { x: 1, y: 0 };
  food = spawnFood();
  score = 0;
  started = false;
  paused = false;
  gameOver = false;

  updateHud();
  setStatus("Press any direction key to start", false);
  draw();
}

function spawnFood() {
  while (true) {
    const next = {
      x: Math.floor(Math.random() * tileCount),
      y: Math.floor(Math.random() * tileCount)
    };

    const onSnake = snake && snake.some((part) => part.x === next.x && part.y === next.y);
    if (!onSnake) {
      return next;
    }
  }
}

function setStatus(text, isOver) {
  statusEl.textContent = text;
  statusEl.classList.toggle("over", Boolean(isOver));
}

function updateHud() {
  scoreEl.textContent = String(score);
  bestEl.textContent = String(best);
}

function drawGrid() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#f2ebde";
  ctx.lineWidth = 1;

  for (let i = 0; i <= tileCount; i += 1) {
    const p = i * gridSize;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(canvas.width, p);
    ctx.stroke();
  }
}

function drawSnake() {
  snake.forEach((part, index) => {
    const isHead = index === 0;
    ctx.fillStyle = isHead ? "#2f7a5f" : "#5faa86";
    ctx.fillRect(part.x * gridSize + 1, part.y * gridSize + 1, gridSize - 2, gridSize - 2);
  });
}

function drawFood() {
  ctx.fillStyle = "#d94f34";
  ctx.beginPath();
  const cx = food.x * gridSize + gridSize / 2;
  const cy = food.y * gridSize + gridSize / 2;
  ctx.arc(cx, cy, gridSize * 0.35, 0, Math.PI * 2);
  ctx.fill();
}

function drawOverlay() {
  if (!paused && !gameOver) {
    return;
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 28px Trebuchet MS";
  ctx.textAlign = "center";

  if (paused) {
    ctx.fillText("Paused", canvas.width / 2, canvas.height / 2);
  }

  if (gameOver) {
    ctx.fillText("Game Over", canvas.width / 2, canvas.height / 2 - 12);
    ctx.font = "bold 18px Trebuchet MS";
    ctx.fillText("Press Enter to restart", canvas.width / 2, canvas.height / 2 + 20);
  }
}

function draw() {
  drawGrid();
  drawSnake();
  drawFood();
  drawOverlay();
}

function step() {
  if (!started || paused || gameOver) {
    draw();
    return;
  }

  direction = queuedDirection;

  const nextHead = {
    x: (snake[0].x + direction.x + tileCount) % tileCount,
    y: (snake[0].y + direction.y + tileCount) % tileCount
  };

  const hitsSelf = snake.some((part) => part.x === nextHead.x && part.y === nextHead.y);
  if (hitsSelf) {
    gameOver = true;
    setStatus("Game over - press Enter to restart", true);
    draw();
    return;
  }

  snake.unshift(nextHead);

  const ateFood = nextHead.x === food.x && nextHead.y === food.y;
  if (ateFood) {
    score += 1;
    if (score > best) {
      best = score;
      saveBest(best);
    }
    food = spawnFood();
    updateHud();
  } else {
    snake.pop();
  }

  draw();
}

function startLoop() {
  if (loopId) {
    clearInterval(loopId);
  }
  loopId = setInterval(step, tickMs);
}

function setDirection(next) {
  if (gameOver) {
    return;
  }

  const isOpposite = direction.x + next.x === 0 && direction.y + next.y === 0;
  if (!isOpposite) {
    queuedDirection = next;
    started = true;
    if (!paused) {
      setStatus("", false);
    }
  }
}

document.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();

  if (key === " ") {
    if (!started || gameOver) {
      return;
    }
    paused = !paused;
    setStatus(paused ? "Paused" : "", false);
    draw();
    return;
  }

  if (key === "enter") {
    resetGame();
    return;
  }

  if (key === "arrowup" || key === "w") {
    setDirection({ x: 0, y: -1 });
  } else if (key === "arrowdown" || key === "s") {
    setDirection({ x: 0, y: 1 });
  } else if (key === "arrowleft" || key === "a") {
    setDirection({ x: -1, y: 0 });
  } else if (key === "arrowright" || key === "d") {
    setDirection({ x: 1, y: 0 });
  }
});

best = loadBest();
updateHud();
resetGame();
startLoop();
