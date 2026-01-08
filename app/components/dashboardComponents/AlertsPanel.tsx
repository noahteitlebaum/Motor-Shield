"use client"

import styles from "./dashboard.module.css";

export default function AlertsPanel(){
    return(
        <>
        <div className={styles.alertsContainer}>
        <p className={styles.alertsTitle}>Alerts</p>
        
        {/*Create horizontal line under the title: */}
        <div style={{
            borderTop: "4px solid #000",
            marginTop: 5,
            marginBottom: 10,
            borderRadius: "10px",
            alignSelf:"center",
            width: "60%",
        }}/>

        <p style={{alignSelf:"center"}}>test text</p>
        </div>
        </>
    );
}