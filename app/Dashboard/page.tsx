"use client"

import CardTest from "../components/cards/CardTest";
import Link from "next/link"
import NavigationBar from "../components/NavigationBar/NavigationBar"
import StatusSection from "../components/dashboardComponents/StatusSection";
import GraphCard from "../components/cards/GraphCard";

export default function Dashboard(){
    return(
        <>
            <NavigationBar/>
            <StatusSection/> {/*the "grey box" section, including the elements inside it*/}


            <h1>This is the dashboard page</h1>
            <div style={{display:"flex", alignContent:"space-between", justifyContent:"left", gap:"5%"}}>
            <GraphCard title="test1" description="desc for graphcard number one." reactGraph={null}/>
            <GraphCard title="test two" description="desc for graphcard number 2." reactGraph={null}/>
            <GraphCard title="test 3" description="desc for graphcard number 3." reactGraph={null}/>
            </div>

            
            {/*Footer is in layout.tsx under the app folder */}
        </>
    )
}
