"use client"

import NavButton from "./NavButton"
import {useEffect, useState} from 'react'

export default function NavigationBar(){

    return(
        <div>

            <NavButton name="Learn More" link="/LearnMore" state={true}/>
        </div>
    )
}