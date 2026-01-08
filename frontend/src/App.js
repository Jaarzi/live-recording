// import React, { useState } from 'react';
// import { ReactMic } from 'react-mic';
// import './App.css';

// function App() {
//   const [record, setRecord] = useState(false);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const startRecording = () => {
//     setRecord(true);
//     setResult(null);
//   };

//   const stopRecording = () => {
//     setRecord(false);
//   };

//   const onStop = async (recordedBlob) => {
//     setLoading(true);
    
//     const formData = new FormData();
//     formData.append('audio', recordedBlob.blob, 'recording.wav');
    
//     try {
//       const response = await fetch('http://localhost:5000/predict', {
//         method: 'POST',
//         body: formData
//       });
      
//       if (!response.ok) throw new Error('Backend error');
//       const data = await response.json();
      
//       console.log('✅ Backend response:', data);
//       setResult(data);
      
//     } catch (error) {
//       console.error('❌ Error:', error);
//       setResult({ error: 'Backend failed. Run: python app.py' });
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1>🛡️ Deepfake Audio Detector</h1>
//         <p>Record 2+ seconds → AI analyzes instantly</p>
        
//         <div className="mic-container">
//           <ReactMic
//             record={record}
//             className="sound-wave"
//             onStop={onStop}
//             visualSetting="sinewave"
//             strokeColor={record ? "#00ff41" : "#666666"}
//             backgroundColor="#1a1a1a"
//             mimeType="audio/wav"           // 🎯 FIXED!
//             audioBitsPerSample={16}        // 🎯 FIXED!
//           />
//         </div>

//         <button 
//           className="analyze-btn"
//           onClick={record ? stopRecording : startRecording}
//           disabled={loading}
//         >
//           {loading ? '🤖 AI Analyzing...' : 
//            (record ? '⏹️ STOP & ANALYZE' : '🎙️ START RECORDING')}
//         </button>

//         {result && !loading && (
//           <div className="result-container">
//             {result.error ? (
//               <div className="error">
//                 <h2>❌ ERROR</h2>
//                 <p>{result.error}</p>
//               </div>
//             ) : (
//               <div className={`success ${result.prediction}`}>
//                 <h2>{result.prediction}</h2>
//                 <p>Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
//               </div>
//             )}
//           </div>
//         )}
//       </header>
//     </div>
//   );
// }

// export default App;




import React, { useState, useRef } from 'react';
import { ReactMic } from 'react-mic';
import './App.css';

function App() {
  const [record, setRecord] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const startTimeRef = useRef(0);

  const startRecording = () => {
    startTimeRef.current = Date.now();
    setRecord(true);
    setResult(null);
  };

  const stopRecording = () => {
    setRecord(false);
  };

  const onData = () => {
    // Visual feedback only
  };

  const onStop = async (recordedBlob) => {
    setLoading(true);
    
    const duration = (Date.now() - startTimeRef.current) / 1000;
    console.log(`🎤 Recorded ${duration.toFixed(1)}s`);
    
    if (duration < 3) {
      setResult({ error: `Record at least 3 seconds! (${duration.toFixed(1)}s)` });
      setLoading(false);
      return;
    }
    
    const formData = new FormData();
    formData.append('audio', recordedBlob.blob, 'live_recording.wav');
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Backend failed. Run: python app.py' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🛡️ Live Deepfake Detector</h1>
        <p>🎙️ Record LIVE voice → AI detects deepfake</p>
        
        <div className="mic-container">
          <ReactMic
            record={record}
            className="sound-wave"
            onStop={onStop}
            onData={onData}
            visualSetting="sinewave"
            strokeColor={record ? "#00ff41" : "#666666"}
            backgroundColor="#1a1a1a"
            mimeType="audio/wav"
            audioBitsPerSample={16}
          />
          {record && (
            <div className="duration">
              ⏱️ {((Date.now() - startTimeRef.current) / 1000).toFixed(1)}s
            </div>
          )}
        </div>

        <button 
          className="analyze-btn"
          onClick={record ? stopRecording : startRecording}
          disabled={loading}
        >
          {loading ? '🤖 AI Analyzing...' : 
           (record ? '⏹️ STOP & ANALYZE' : '🎙️ START RECORDING')}
        </button>

        {result && !loading && (
          <div className={`result-container ${result.error ? 'error' : result.prediction?.toLowerCase()}`}>
            {result.error ? (
              <div className="error">
                <h2>❌ ERROR</h2>
                <p>{result.error}</p>
              </div>
            ) : (
              <div className={`success ${result.prediction.toLowerCase()}`}>
                <h2>{result.prediction}</h2>
                <p>Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
