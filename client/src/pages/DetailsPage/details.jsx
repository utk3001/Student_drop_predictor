import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import styles from "./details.module.css";
import axios from "axios";

const DetailsPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const rollNumber = location.state?.rollNumber;
    const API_BASE_URL = process.env.REACT_APP_API_URL;

    const [studentData, setStudentData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);
    const [error, setError] = useState("");

    const DISPLAY_FIELDS = [
        "Marital Status",
        "Application Mode",
        "Daytime/evening attendance",
        "Previous qualification",
        "Previous qualification (grade)",
        "Age at enrollment",
        "Educational special needs",
        "Course",
        "Admission grade",
        "International",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)"
    ];

    const FIELD_LABELS = {
        "Daytime/evening attendance": "Daytime/Evening Attendance",
        "Previous qualification": "Previous Qualification",
        "Previous qualification (grade)":"Previous Qualification Grade",
        "Age at enrollment":  "Age at Enrollment",
        "Educational special needs": "Special Needs",
        "Admission grade":"Admission Grade",
        "International": "International Student",
        "Curricular units 1st sem (enrolled)":"1st Sem Enrolled Units",
        "Curricular units 2nd sem (enrolled)":"2nd Sem Enrolled Units",
        "Curricular units 1st sem (grade)":"1st Sem Grade",
        "Curricular units 2nd sem (grade)":"2nd Sem Grade",
};

    useEffect(() => {
        const fetchStudentData = async () => {
            try {
                const response = await axios.get(
                    `${API_BASE_URL}/get_student/${rollNumber}`
                );
                setStudentData(response.data);
            } catch (err) {
                setError(err.response?.data?.error || "Failed to fetch student");
            } finally {
                setLoading(false);
            }
        };
        fetchStudentData();
    }, [rollNumber]);

    const handlePredictClick = async () => {
        if (!studentData) return;
        setPredicting(true);
        setPrediction(null);
        setError("");

        try {
            const response = await axios.post(
                `${API_BASE_URL}/predict`,
                studentData
            );
            console.log("Prediction response:", response.data);
            setPrediction({
                outcome:
                    response.data.prediction === 1 ? "Likely to Dropout" : "Likely to Graduate",
                confidence: response.data.confidence,
                overall_accuracy: response.data.overall_accuracy,
                fairness_metrics: response.data.fairness_metrics || [],
                justification: response.data.justification || [],
            });
        } catch (err) {
            console.error("Prediction error:", err);
            setError(err.response?.data?.error || "Prediction failed");
        } finally {
            setPredicting(false);
        }
    };

    const handleBack = () => navigate("/landing");

    if (loading) {
        return (
            <div className={styles.loadingContainer}>
                <p className={styles.loadingText}>Fetching student data...</p>
            </div>
        );
    }

    if (!studentData) {
        return (
            <div className={styles.errorContainer}>
                <p className={styles.errorText}>No student data found.</p>
            </div>
        );
    }

    return (
        <div className={styles.pageContainer}>
            <button className={styles.backButton} onClick={handleBack}>
                ← Back
            </button>

            <header className={styles.header}>
                <h2 className={styles.brand}>Student's Dropout Predictor</h2>
                <p className={styles.tagline}>Detailed Data View</p>
            </header>

            <main className={styles.card}>
                <h1 className={styles.title}>
                    Student Details — Roll No. {rollNumber}
                </h1>

                <div className={styles.dataGrid}>
                    {Object.entries(studentData)
                        .filter(([key]) => DISPLAY_FIELDS.includes(key))
                        .map(([key, value]) => (
                            <div key={key} className={styles.dataRow}>
                                <span className={styles.label}>
                                    {FIELD_LABELS[key] || key}
                                </span>
                                <span className={styles.value}>{value}</span>
                            </div>
                        ))}
                </div>

                <button
                    className={styles.predictButton}
                    onClick={handlePredictClick}
                    disabled={predicting}
                >
                    {predicting ? "Predicting..." : "Predict Outcome"}
                </button>

                {error && <p className={styles.errorText}>{error}</p>}

                {prediction && (
                    <div className={styles.predictionContainer}>
                        {/* Prediction Box */}
                        <div className={styles.predictionBox}>
                            <h3 className={styles.predictionTitle}>Prediction Result</h3>
                            <p className={styles.predictionText}>
                                Outcome: <strong>{prediction.outcome}</strong>
                            </p>
                            <p className={styles.predictionText}>
                                Confidence: <strong>{prediction.confidence}%</strong>
                            </p>
                            {prediction.overall_accuracy && (
                                <p className={styles.predictionText}>
                                    Overall Accuracy: <strong>{prediction.overall_accuracy}</strong>
                                </p>
                            )}
                        </div>

                        {/* Justification */}
                        {prediction.justification?.length > 0 && (
                            <div className={styles.justificationBox}>
                                <h3 className={styles.predictionTitle}>Why the model predicted this?</h3>
                                <ul>
                                    {prediction.justification.map((reason, index) => (
                                        <li key={index}>{reason}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {/* Fairness Metrics */}
                        {prediction.fairness_metrics?.length > 0 && (
                            <div className={styles.fairnessBox}>
                                <h3 className={styles.predictionTitle}>Fairness Metrics after Bias Mitigation</h3>
                                {prediction.fairness_metrics.map((metric, index) => (
                                    <div key={index} className={styles.fairnessRow}>
                                        <p>
                                            <strong>{metric.group}</strong>: {metric.difference_in_dropout_rate}
                                        </p>
                                        <p className={styles.fairnessInterpretation}>{metric.interpretation}</p>
                                    </div>
                                ))}
                            </div>
                        )}



                        <button className={styles.anotherButton} onClick={handleBack}>
                            Predict for Another Student
                        </button>
                    </div>
                )}

                <p className={styles.note}>
                    *Predictions are based on current academic indicators and model training data.
                </p>
            </main>

            <footer className={styles.footer}>
                <p>© 2025 AI System | IIT Kanpur</p>
            </footer>
        </div>
    );
};

export default DetailsPage;
