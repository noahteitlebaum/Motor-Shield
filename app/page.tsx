"use client"

import {useEffect} from 'react'
import { useRouter } from 'next/navigation'

import './globals.css'

function App(){
  const router = useRouter()
    useEffect(() => {
      
      router.push('/Dashboard')
    })

    return <p>Redirecting...</p>
}

export default App