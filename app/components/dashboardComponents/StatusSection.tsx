//this is the grey box that encapsulates this section of the webpage. See Figma
"use client"

import styles from "./dashboard.module.css";
import AlertsPanel from "./AlertsPanel";
import MotorStatus from "./MotorStatus";

export default function StatusSection(){
    return(
        <div className={styles.statusBox}>
            test content
            <AlertsPanel/>
        </div>
    );
}