import React, { useState, useEffect } from "react";
import styles from "./landingPage.module.css";
import { useNavigate } from "react-router-dom";

const LandingPage = () => {
    const navigate = useNavigate();

    // ✅ Initialize from localStorage only for economic parameters
    const getInitialFormData = () => {
        const cached = localStorage.getItem("economicParams");
        if (cached) {
            try {
                const parsed = JSON.parse(cached);
                return {
                    rollNumber: "", // never cached
                    unemploymentRate: parsed.unemploymentRate ?? 7.5,
                    inflationRate: parsed.inflationRate ?? 6.2,
                    GDP: parsed.GDP ?? 5.2,
                };
            } catch {
                return {
                    rollNumber: "",
                    unemploymentRate: 7.5,
                    inflationRate: 6.2,
                    GDP: 5.2,
                };
            }
        }
        return {
            rollNumber: "",
            unemploymentRate: 7.5,
            inflationRate: 6.2,
            GDP: 5.2,
        };
    };

    const [editable, setEditable] = useState(false);
    const [formData, setFormData] = useState(getInitialFormData);

    // Cache only economic values when they change
    useEffect(() => {
        const { unemploymentRate, inflationRate, GDP } = formData;
        localStorage.setItem(
            "economicParams",
            JSON.stringify({ unemploymentRate, inflationRate, GDP })
        );
    }, [formData.unemploymentRate, formData.inflationRate, formData.GDP]);

    const handleBack = () => navigate("/");

    const handleChange = (e) => {
        let { name, value } = e.target;

        // Validation limits
        if (name === "unemploymentRate") {
            if (value > 15) value = 15;
            if (value < 0) value = 0;
        }
        if (name === "inflationRate") {
            if (value > 10) value = 10;
            if (value < 0) value = 0;
        }

        setFormData({ ...formData, [name]: value });
    };

    const toggleEditable = () => setEditable(!editable);

    const handleSubmit = (e) => {
        e.preventDefault();
        navigate("/details", { state: { rollNumber: formData.rollNumber } });
    };

    return (
        <div className={styles.pageContainer}>
            <button className={styles.backButton} onClick={handleBack}>
                ← Back
            </button>

            <header className={styles.header}>
                <h2 className={styles.brand}>Student's Dropout Predictor</h2>
                <p className={styles.tagline}>Enter details to generate prediction</p>
            </header>

            <main className={styles.card}>
                <h1 className={styles.title}>Enter Roll Number</h1>

                <form onSubmit={handleSubmit}>
                    <div className={styles.inputGroup}>
                        <input
                            type="text"
                            name="rollNumber"
                            value={formData.rollNumber}
                            onChange={handleChange}
                            placeholder="Enter Roll Number"
                            required
                            className={styles.inputField}
                        />
                    </div>

                    <div className={styles.defaultsBox}>
                        <p><strong>Default Parameters:</strong></p>
                        <p>Unemployment Rate (%): {formData.unemploymentRate}</p>
                        <p>Inflation Rate (%): {formData.inflationRate}</p>
                        <p>GDP (trillion): {formData.GDP}</p>

                        <button
                            type="button"
                            className={styles.toggleButton}
                            onClick={toggleEditable}
                        >
                            {editable ? "Lock Defaults" : "Edit Parameters"}
                        </button>
                    </div>

                    {editable && (
                        <div className={styles.editSection}>
                            <label>
                                Unemployment Rate (%):
                                <input
                                    type="number"
                                    step="0.01"
                                    min="0"
                                    max="15"
                                    name="unemploymentRate"
                                    value={formData.unemploymentRate}
                                    onChange={handleChange}
                                    required
                                    className={styles.editInput}
                                />
                            </label>

                            <label>
                                Inflation Rate (%):
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="10"
                                    name="inflationRate"
                                    value={formData.inflationRate}
                                    onChange={handleChange}
                                    required
                                    className={styles.editInput}
                                />
                            </label>

                            <label>
                                GDP (trillion):
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="10"
                                    name="GDP"
                                    value={formData.GDP}
                                    onChange={handleChange}
                                    required
                                    className={styles.editInput}
                                />
                            </label>
                        </div>
                    )}

                    <button type="submit" className={styles.predictButton}>
                        Fetch Details
                    </button>
                </form>

                <p className={styles.note}>
                    *Default values are derived from average historical data. 
                    <br/>
                    Update if specific data is available.
                    <br/>
                    Your last entered economic values will be saved automatically for convenience.
                </p>
            </main>

            <footer className={styles.footer}>
                <p>© 2025 AI System | IIT Kanpur</p>
            </footer>
        </div>
    );
};

export default LandingPage;
