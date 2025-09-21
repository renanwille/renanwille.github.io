import React, { useState } from 'react';

const initialImage = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0],
];

const kernel = [
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0],
];

function applyDilatation(image, kernel) {
  // Simple binary dilatation for demonstration
  const n = image.length;
  const m = image[0].length;
  const k = kernel.length;
  const offset = Math.floor(k / 2);
  const output = Array.from({ length: n }, () => Array(m).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let dilate = 0;
      for (let ki = 0; ki < k; ki++) {
        for (let kj = 0; kj < k; kj++) {
          const ni = i + ki - offset;
          const nj = j + kj - offset;
          if (
            ni >= 0 &&
            ni < n &&
            nj >= 0 &&
            nj < m &&
            kernel[ki][kj] === 1 &&
            image[ni][nj] === 1
          ) {
            dilate = 1;
          }
        }
      }
      output[i][j] = dilate;
    }
  }
  return output;
}

function renderHoverMatrix(matrix, hoveredCell) {
  // Print at the console hoveredCell output
  console.log('Hovered Cell:', hoveredCell?.i, hoveredCell?.j);
  return (
    <table style={{ borderCollapse: 'collapse' }}>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td
                key={j}
                style={{
                  width: 20,
                  height: 20,
                  border: '1px solid #ccc',
                  background:
                    hoveredCell?.i === i && hoveredCell?.j === j
                      ? '#f00'
                      : cell
                      ? '#333'
                      : '#fff',
                  transition: 'background 0.2s',
                  cursor: 'default',
                }}
              />
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function renderMatrix(matrix, kernelSize = 0) {
  const n = matrix.length;
  const m = matrix[0].length;
  return (
    <table style={{ borderCollapse: 'collapse' }}>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td
                key={j}
                style={{
                  width: 20,
                  height: 20,
                  border: '1px solid #ccc',
                  background: cell
                    ? '#333'
                    : '#fff',
                  transition: 'background 0.2s',
                  cursor: 'default',
                }}
              />
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function DilatationVisualization() {
  const [output, setOutput] = useState(applyDilatation(initialImage, kernel));
  const [hoveredCell, setHoveredCell] = useState(null);

  const kernelSize = kernel.length;

  function handleCellEnter(i, j) {
    setHoveredCell({ i, j });
  }

  function handleCellLeave() {
    setHoveredCell(null);
  }

  function renderOriginalImage() {
    return (
      <table style={{ borderCollapse: 'collapse' }}>
        <tbody>
          {initialImage.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => {
                // Determine if this cell should be highlighted
                let highlightCells = new Set();
                if (hoveredCell) {
                  const offset = Math.floor(kernelSize / 2);
                  for (let ki = 0; ki < kernelSize; ki++) {
                    for (let kj = 0; kj < kernelSize; kj++) {
                      const ni = hoveredCell.i + ki - offset;
                      const nj = hoveredCell.j + kj - offset;
                      if (ni >= 0 && ni < initialImage.length && nj >= 0 && nj < initialImage[0].length) {
                        highlightCells.add(`${ni},${nj}`);
                      }
                    }
                  }
                }
                return (
                  <td
                    key={j}
                    style={{
                      width: 20,
                      height: 20,
                      border: '1px solid #ccc',
                      outline: highlightCells.has(`${i},${j}`)
                        ? '3px solid #f00' : '1px solid #ccc',
                      outlineOffset: '-1px',
                      background: cell
                        ? '#333'
                        : '#fff',
                      transition: 'background 0.2s',
                      cursor: 'pointer',
                    }}
                    
                    onMouseEnter={() => handleCellEnter(i, j)}
                    onMouseLeave={handleCellLeave}
                  />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '32px', alignItems: 'flex-start' }}>
      <div>
        <h3>Original Image</h3>
        {renderOriginalImage()}
      </div>
      <div>
        <h3>Kernel</h3>
        {renderMatrix(kernel)}
      </div>
      <div>
        <h3>Dilatation Output</h3>
        {renderHoverMatrix(output, hoveredCell)}
      </div>
    </div>
  );
}