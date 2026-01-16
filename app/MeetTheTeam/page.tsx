import NavigationBar from "../components/NavigationBar/NavigationBar"
import "./MeetTheTeam.css"

type TeamMember = {
  name: string
  role: string
  bio: string
  imgSrc: string
}

const team: TeamMember[] = [
  {
    name: "Noah Teitlebaum",
    role: "Team Lead",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "Vedanshi Parekh",
    role: "Team Lead",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "Pratik Gupta",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Daniel He",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Adison Seo",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Krish Singh",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Aidan Naccarato",
    role: "Frontend",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Baichaun Ren",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Justin Jubinville",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Zayan Suri",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
]

export function MeetTheTeam(){
    return(
        <>
            <NavigationBar/>
            
            <div className="page-bg">
                <div className="panel">
                <div className="panel-header">
                    <h1 className="panel-title">Meet the Team</h1>
                    <p className="panel-subtitle">
                    The people building MotorShield.
                    </p>
                </div>

                <div className="team-grid">
                    {team.map((m) => (
                    <div key={m.name} className="team-card">
                        <div className="avatar-outer">
                        <div className="avatar-inner">
                            <img src={m.imgSrc} alt={m.name} />
                        </div>
                        </div>

                        <div className="team-text">
                        <h3 className="team-name">{m.name}</h3>
                        <p className="team-role">{m.role}</p>
                        <p className="team-bio">{m.bio}</p>
                        </div>
                    </div>
                    ))}
                </div>
                </div>
            </div>
            
        </>
    )
}

export default MeetTheTeam