import React from "react";
import styles from "./homePage.module.css";
import { useNavigate } from "react-router-dom";

const HomePage = () => {

  const Navigate = useNavigate();
  const handlePredictClick = () => {
    Navigate("/landing");
  };

  return (
    <div className={styles.pageContainer}>
      <header className={styles.header}>
        <h2 className={styles.brand}>Student's Dropout Predictor</h2>
        <p className={styles.tagline}>
          uses AI to enhance educational outcomes
        </p>
      </header>

      <main className={styles.card}>
        <h1 className={styles.title}>Predict Student Outcomes</h1>
        <p className={styles.explanation}>
          This interface allows you to input student-related attributes and view the
          model’s prediction on academic success. It also justifies its prediction to ensure responsible and transparent AI usage.
        </p>

        <button className={styles.predictButton} onClick={handlePredictClick}>
          Run Prediction
        </button>

        <p className={styles.note}>
          *Predictions are based on historical data and may not be fully accurate.<br /> (Current Accuracy: 88.1%)
        </p>
      </main>

      <footer className={styles.footer}>
        <p>© 2025 AI System | IIT Kanpur</p>
      </footer>
    </div>
  );
};

export default HomePage;
