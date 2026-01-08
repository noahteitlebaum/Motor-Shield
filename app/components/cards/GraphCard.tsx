"use client";

import { ReactNode } from "react";
import styles from "./GraphCard.module.css";
import { useState } from "react";

type props = {
    title: string;
    description: string;
    reactGraph: ReactNode;
};

export default function GraphCard({title, description, reactGraph}: props) {
    const [isFlipped, setFlipped] = useState(false);
    const handleFlip = () => {
        setFlipped(!isFlipped);
    };

   return(
    <div className={styles.graphContainer}>
        <div className={`${styles.card} ${isFlipped ? styles.flipped : ""}`}>
        {/*split into front face and back face*/}

        {/*FRONT FACE*/}
        <div className={styles.frontFace}>
            <button className={styles.flipButton} onClick={handleFlip}> <img src='/swapArrowCrop.png' alt="flip"/> </button>

            <div className={styles.titleBox}>
            <h3 className={styles.title}>{title}</h3>
            </div>

            <span className={styles.label}>{reactGraph}</span>
        </div>

        {/*BACK FACE*/}
        <div className={styles.backFace}>
            <button className={styles.flipButton} onClick={handleFlip}> <img src="/swapArrowCrop.png" alt="flip"/> </button>

            <div className={styles.titleBox}>
            <h3 className={styles.title}>{title}</h3>
            </div>

            <div className={styles.backFaceBox}>
                <p className={styles.description}>{description}</p>
            </div>
            
        </div>

        </div>
    </div>
    );
}