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
    name: "person1",
    role: "Founder / Backend",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "person 2",
    role: "Frontend / UX",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "person 3",
    role: "ML / Data",
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