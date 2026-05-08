import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

export default function App() {
  const videoRef = useRef(null);
  
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [currentWord, setCurrentWord] = useState("");
  const [sentence, setSentence] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [handBox, setHandBox] = useState(null);
  
  // Use refs for local tracking within the animation frame to avoid stale state closures
  const currentWordRef = useRef("");
  
  // Track consecutive identical predictions for auto-typing
  const predictionBuffer = useRef([]);
  const framesHeld = useRef(0);
  const lastStable = useRef(null);
  
  // Set slower reading limit (higher frames = longer wait before typing)
  const MAX_FRAMES = 150;

  function handleClear() {
    setSentence("");
    setCurrentWord("");
    currentWordRef.current = "";
    predictionBuffer.current.length = 0;
    framesHeld.current = 0;
    setProgress(0);
  }

  useEffect(() => {
    let landmarker = null;
    let keepRunning = true;
    let activeStream = null;

    async function init() {
      if (!isCameraActive) return;
      setLoading(true);
      try {
        // 1. Load MediaPipe Hand Landmarker via WASM
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        setLoading(false);

        // 2. Open User's Camera
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        activeStream = stream;
        
        // If component unmounted or camera toggled off while waiting, immediately stop tracks
        if (!keepRunning) {
            stream.getTracks().forEach(track => track.stop());
            if (landmarker) landmarker.close();
            return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        if (videoRef.current) {
          videoRef.current.onloadeddata = () => {
            if (!keepRunning) return;
            videoRef.current.play().catch(e => console.log("Play interrupted", e));
            predictLoop();
          };
        }
      } catch (err) {
        console.error("Camera or MediaPipe initialization failed.", err);
        setLoading(false);
      }
    }

    async function predictLoop() {
      if (!keepRunning || !videoRef.current || !landmarker) return;

      const startTimeMs = performance.now();
      const results = landmarker.detectForVideo(videoRef.current, startTimeMs);

      if (results.landmarks && results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        
        // Compute bounding box for UI rendering
        const xs = hand.map(lm => lm.x);
        const ys = hand.map(lm => lm.y);
        setHandBox({ minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) });
        
        // 3. Normalize into Relative Coordinates (Same as Python Script)
        const baseX = hand[0].x;
        const baseY = hand[0].y;
        const baseZ = hand[0].z;

        const relativeLandmarks = [];
        hand.forEach((lm) => {
          relativeLandmarks.push(lm.x - baseX, lm.y - baseY, lm.z - baseZ);
        });

        // 4. Send to Django Backend
        try {
          const res = await fetch("http://localhost:8000/api/predict/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ landmarks: relativeLandmarks })
          });
          const data = await res.json();
          
          if (data.prediction) {
            if (data.confidence !== undefined) {
               setConfidence(data.confidence);
            }
            handlePredictionState(data.prediction);
          }
        } catch (error) {
          // Ignore network errs during loop
        }
      } else {
        setPrediction("nothing");
        setConfidence(0);
        setHandBox(null);
        predictionBuffer.current.length = 0;
        framesHeld.current = 0;
        lastStable.current = null;
        setProgress(0);
      }

      requestAnimationFrame(predictLoop);
    }

    if (isCameraActive) {
      init();
    }
    
    return () => { 
        keepRunning = false; 
        
        // Stop all tracks directly from the tracked stream variable
        if (activeStream) {
            activeStream.getTracks().forEach(track => track.stop());
        }
        
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        
        if (landmarker) {
            landmarker.close();
        }
        setHandBox(null);
        setPrediction("");
        setProgress(0);
    };
  }, [isCameraActive]);

  function handlePredictionState(pred) {
      setPrediction(pred);
      predictionBuffer.current.push(pred);
      if (predictionBuffer.current.length > 15) predictionBuffer.current.shift();
      
      // Calculate Most Frequent Prediction
      const freqMap = {};
      predictionBuffer.current.forEach(p => freqMap[p] = (freqMap[p] || 0) + 1);
      const stablePred = Object.keys(freqMap).reduce((a, b) => freqMap[a] > freqMap[b] ? a : b);

      if (stablePred === lastStable.current) {
          framesHeld.current += 1;
      } else {
          framesHeld.current = 0;
          lastStable.current = stablePred;
      }

      // Update progress bar UI smoothly
      setProgress(Math.max(0, Math.min(100, (framesHeld.current / MAX_FRAMES) * 100)));

      // Auto-type when held for enough frames
      if (framesHeld.current >= MAX_FRAMES) {
          if (stablePred === 'space') {
              // Word is complete, add space. Capture current word first!
              const wordToAdd = currentWordRef.current;
              
              if (wordToAdd.trim() !== "") {
                  // Ping backend to autocorrect spelling using TextBlob
                  fetch("http://localhost:8000/api/autocorrect/", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ text: wordToAdd })
                  })
                  .then(r => r.json())
                  .then(data => {
                      const finalWord = data.corrected || wordToAdd;
                      setSentence(s => s + finalWord + " ");
                  })
                  .catch(err => {
                      setSentence(s => s + wordToAdd + " ");
                  });
              } else {
                  setSentence(s => s + " ");
              }
              
              currentWordRef.current = "";
              setCurrentWord("");
          } else if (stablePred === 'del') {
              currentWordRef.current = currentWordRef.current.slice(0, -1);
              setCurrentWord(currentWordRef.current);
          } else if (stablePred !== 'nothing') {
              currentWordRef.current += stablePred;
              setCurrentWord(currentWordRef.current);
          }
          framesHeld.current = -30; // Double cooldown to prevent rapid typing
          setProgress(0);
          predictionBuffer.current.length = 0;
      }
  }

  return (
    <div style={{ fontFamily: "'Segoe UI', Roboto, Helvetica, Arial, sans-serif", minHeight: "100vh", backgroundColor: "#0f0f1a", color: "#ffffff", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      
      {/* Top Navbar */}
      <nav style={{ 
        display: "flex", justifyContent: "space-between", alignItems: "center", 
        padding: "15px 40px", backgroundColor: "#1a1a2e", boxShadow: "0 4px 15px rgba(0,0,0,0.5)",
        borderBottom: "1px solid #33334d"
      }}>
        <h1 style={{ margin: 0, color: "#b388ff", textTransform: "uppercase", letterSpacing: "3px", textShadow: "0 0 10px rgba(179, 136, 255, 0.5)", fontSize: "1.5rem" }}>
          ASL Neural Dashboard
        </h1>
        <button 
          onClick={() => setIsCameraActive(!isCameraActive)}
          style={{
            padding: "10px 25px",
            backgroundColor: isCameraActive ? "transparent" : "#b388ff",
            border: isCameraActive ? "2px solid #ff4d4d" : "2px solid #b388ff",
            color: isCameraActive ? "#ff4d4d" : "#0f0f1a",
            borderRadius: "8px",
            fontSize: "1rem",
            cursor: "pointer",
            fontWeight: "bold",
            textTransform: "uppercase",
            transition: "all 0.3s ease",
            boxShadow: isCameraActive ? "none" : "0 0 10px rgba(179, 136, 255, 0.5)"
          }}
        >
          {isCameraActive ? "Stop Camera" : "Start Camera"}
        </button>
      </nav>

      <div style={{ display: "flex", gap: "30px", padding: "40px", flexWrap: "wrap", justifyContent: "center", flex: 1 }}>
        
        {/* Camera Output */}
        <div style={{ flex: "1", minWidth: "400px", maxWidth: "600px", position: "relative" }}>
           <div style={{ 
              position: "relative", width: "100%", borderRadius: "16px", overflow: "hidden",
              boxShadow: "0 0 20px rgba(138, 43, 226, 0.4)", backgroundColor: "#000", border: "2px solid #8a2be2",
              aspectRatio: "4/3", display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            {!isCameraActive && !loading && (
              <p style={{ color: "#a0a0b5", fontSize: "1.2rem", letterSpacing: "1px" }}>CAMERA IS OFFLINE</p>
            )}
            {loading && (
              <p style={{ color: "#00ffff", fontSize: "1.2rem", animation: "blink 1.5s infinite" }}>Initializing Neural Architecture...</p>
            )}
            
            <video 
              ref={videoRef} 
              style={{ 
                width: "100%", height: "100%", objectFit: "cover",
                transform: "scaleX(-1)", 
                display: isCameraActive && !loading ? "block" : "none"
              }} 
            />

            {/* Tracking Overlay Box with Loading Bar */}
            {isCameraActive && handBox && prediction && prediction !== "nothing" && (
                <div style={{
                    position: "absolute",
                    top: `${handBox.minY * 100}%`,
                    left: `${(1 - handBox.maxX) * 100}%`,
                    width: `${(handBox.maxX - handBox.minX) * 100}%`,
                    height: `${(handBox.maxY - handBox.minY) * 100}%`,
                    border: "3px solid #00ffff",
                    borderRadius: "8px",
                    boxShadow: "0 0 15px rgba(0, 255, 255, 0.5), inset 0 0 10px rgba(0, 255, 255, 0.3)",
                    pointerEvents: "none",
                    transition: "all 0.1s ease-out"
                }}>
                    <div style={{
                        position: "absolute", top: "-38px", left: "-3px",
                        backgroundColor: "#00ffff", color: "#000",
                        padding: "4px 12px", borderRadius: "4px", fontWeight: "bold",
                        display: "flex", alignItems: "center", gap: "10px",
                        boxShadow: "0 2px 10px rgba(0, 255, 255, 0.5)",
                        fontSize: "1.1rem"
                    }}>
                        <span>{prediction}</span>
                        <div style={{ width: "40px", height: "6px", background: "rgba(0,0,0,0.3)", borderRadius: "3px", overflow: "hidden" }}>
                            <div style={{ width: `${progress}%`, height: "100%", background: "#0f0f1a", transition: "width 0.1s linear" }} />
                        </div>
                    </div>
                </div>
            )}
          </div>
        </div>
        
        {/* Sidebar Data */}
        <div style={{ flex: "1", fontSize: "1.2rem", padding: "30px", backgroundColor: "#1a1a2e", borderRadius: "16px", border: "1px solid #33334d", boxShadow: "inset 0 0 20px rgba(0,0,0,0.5)", display: "flex", flexDirection: "column" }}>
          <h2 style={{ color: "#00ffff", textTransform: "uppercase", fontSize: "1.2rem", letterSpacing: "2px", margin: "0 0 15px 0" }}>Live Telemetry</h2>
          <hr style={{ borderColor: "#8a2be2", opacity: 0.3, marginBottom: "25px", width: "100%" }} />
          
          <p style={{ margin: "0 0 40px 0", display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ color: "#a0a0b5" }}>DETECTING:</span> 
            <span style={{ padding: "8px 20px", backgroundColor: "rgba(179, 136, 255, 0.1)", border: "1px solid #b388ff", color: "#b388ff", borderRadius: "8px", fontWeight: "bold" }}>
              {prediction || "Scanning..."}
            </span>
            {prediction && prediction !== "nothing" && (
                <span style={{ 
                    fontSize: "0.85rem", 
                    color: confidence > 80 ? "#00ff00" : (confidence > 50 ? "#ffcc00" : "#ff4d4d"), 
                    marginLeft: "10px" 
                }}>
                    ACCURACY: {confidence.toFixed(1)}%
                </span>
            )}
          </p>
          
          <div style={{ marginBottom: "40px" }}>
              <p style={{ color: "#a0a0b5", marginBottom: "10px", fontSize: "0.9rem", textTransform: "uppercase" }}>Current Buffer:</p>
              <div style={{ fontSize: "2.5rem", color: "#ffffff", minHeight: "50px", borderBottom: "2px solid rgba(0, 255, 255, 0.3)", textShadow: "0 0 8px rgba(255, 255, 255, 0.3)" }}>
                  {currentWord}
                  <span style={{ opacity: 0.5, animation: "blink 1s infinite" }}>|</span>
              </div>
          </div>

          <div style={{ flex: 1 }}>
              <p style={{ color: "#a0a0b5", marginBottom: "10px", fontSize: "0.9rem", textTransform: "uppercase" }}>Output String:</p>
              <div style={{ fontSize: "1.8rem", color: "#00ffff", minHeight: "40px", textShadow: "0 0 10px rgba(0, 255, 255, 0.4)" }}>
                  {sentence}
              </div>
          </div>
          
          <button 
            onClick={handleClear}
            style={{
              marginTop: "20px",
              padding: "10px 20px",
              backgroundColor: "transparent",
              border: "2px solid #ff4d4d",
              color: "#ff4d4d",
              borderRadius: "8px",
              fontSize: "1rem",
              cursor: "pointer",
              fontWeight: "bold",
              textTransform: "uppercase",
              transition: "all 0.3s ease"
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#ff4d4d";
              e.target.style.color = "#ffffff";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "transparent";
              e.target.style.color = "#ff4d4d";
            }}
          >
            Clear All
          </button>
        </div>
      </div>
      
      <style>{`
        @keyframes blink { 50% { opacity: 0 } }
      `}</style>
    </div>
  );
}
