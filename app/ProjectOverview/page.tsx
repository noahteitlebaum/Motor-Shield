import NavigationBar from "../components/NavigationBar/NavigationBar";

export function ProjectOverview() {
  return (
    <>
      <NavigationBar />

      <div style={{ 
        minHeight: '100vh', 
        backgroundColor: '#ffffff', 
        padding: '2rem 1.5rem',
        fontFamily: 'var(--font-geist-sans), Arial, Helvetica, sans-serif'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
          {/* Header */}
          <div style={{ marginBottom: '3rem' }}>
            <p style={{
              fontSize: '0.875rem',
              fontWeight: '600',
              letterSpacing: '0.05em',
              color: '#6299BB',
              textTransform: 'uppercase',
              marginBottom: '0.5rem'
            }}>
              MotorShield
            </p>

            <h1 style={{
              marginTop: '0.5rem',
              fontSize: '2.5rem',
              fontWeight: '900',
              letterSpacing: '-0.02em',
              color: '#36404D',
              marginBottom: '1rem'
            }}>
              Project Overview
            </h1>

            <p style={{
              marginTop: '1rem',
              maxWidth: '800px',
              fontSize: '1rem',
              lineHeight: '1.75',
              color: '#36404D'
            }}>
              MotorShield is an AI tool that predicts motor failure before it happens.
              Instead of waiting for breakdowns, we use motor signal data to estimate risk
              early. Through this, we help reduce save money by avoiding expensive repairs, and keep systems
              running smoothly.
            </p>
          </div>

          {/* Why it matters */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.5rem',
            marginBottom: '3rem'
          }}>
            <div style={{
              borderRadius: '20px',
              border: '1px solid #d0e5f1',
              backgroundColor: '#F0F0F0',
              padding: '1.5rem',
              boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '700',
                color: '#36404D',
                marginBottom: '0.75rem'
              }}>Less downtime</h3>
              <p style={{
                marginTop: '0.5rem',
                fontSize: '0.9375rem',
                lineHeight: '1.75',
                color: '#36404D'
              }}>
                Unexpected motor failure can shut down an entire operation. MotorShield helps
                catch warning signs early so teams can act before things go down.
              </p>
            </div>

            <div style={{
              borderRadius: '20px',
              border: '1px solid #d0e5f1',
              backgroundColor: '#F0F0F0',
              padding: '1.5rem',
              boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '700',
                color: '#36404D',
                marginBottom: '0.75rem'
              }}>Lower cost</h3>
              <p style={{
                marginTop: '0.5rem',
                fontSize: '0.9375rem',
                lineHeight: '1.75',
                color: '#36404D'
              }}>
                Predicting issues early helps reduce emergency repair costs, delays, and
                part replacement caused by waiting too long.
              </p>
            </div>

            <div style={{
              borderRadius: '20px',
              border: '1px solid #d0e5f1',
              backgroundColor: '#F0F0F0',
              padding: '1.5rem',
              boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '700',
                color: '#36404D',
                marginBottom: '0.75rem'
              }}>Clear decisions</h3>
              <p style={{
                marginTop: '0.5rem',
                fontSize: '0.9375rem',
                lineHeight: '1.75',
                color: '#36404D'
              }}>
                Instead of raw sensor graphs, we summarize motor condition into a simple
                health score and trend that's easy to read fast.
              </p>
            </div>
          </div>

          {/* How it works */}
          <div style={{ marginTop: '3rem' }}>
            <h2 style={{
              fontSize: '2rem',
              fontWeight: '900',
              color: '#36404D',
              marginBottom: '0.75rem'
            }}>How it works</h2>
            <p style={{
              marginTop: '0.5rem',
              maxWidth: '800px',
              fontSize: '0.9375rem',
              lineHeight: '1.75',
              color: '#36404D',
              marginBottom: '2rem'
            }}>
              MotorShield follows a simple pipeline: simulate motor data, train the model,
              validate with sensors, and show results through a dashboard.
            </p>

            {/* Visual Flow Diagram */}
            <div style={{
              marginBottom: '3rem',
              padding: '2rem',
              borderRadius: '20px',
              border: '1px solid #d0e5f1',
              backgroundColor: '#F0F0F0',
              boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
            }}>
              <PipelineDiagram />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <Step
                num="1"
                title="Simulink Simulation"
                desc="We generate motor behavior in a controlled environment. This creates realistic data with both healthy and failing conditions."
                icon={<SimulinkIcon />}
              />

              <Step
                num="2"
                title="Deep Learning Model"
                desc="We train a model to detect patterns that appear before failure happens. Our current best approach is a CNN → LSTM pipeline for time-series data."
                icon={<AIModelIcon />}
              />

              <Step
                num="3"
                title="Microcontroller + Sensors"
                desc="Eventually, we'll collect real motor readings using sensors and a microcontroller, then run predictions in real time to validate performance outside simulation."
                icon={<SensorIcon />}
              />

              <Step
                num="4"
                title="Website Dashboard"
                desc="The web app visualizes motor health, trends, and failure risk in a clean format that's easy to understand."
                icon={<DashboardIcon />}
              />
            </div>
          </div>

          {/* Example output visual */}
          <div style={{
            marginTop: '3rem',
            borderRadius: '20px',
            border: '1px solid #d0e5f1',
            backgroundColor: '#F0F0F0',
            padding: '2rem',
            boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
              marginBottom: '1.5rem'
            }}>
              <div>
                <p style={{
                  fontSize: '0.875rem',
                  color: '#6299BB',
                  fontWeight: '600',
                  marginBottom: '0.25rem'
                }}>Example Output</p>
                <h3 style={{
                  marginTop: '0.25rem',
                  fontSize: '1.5rem',
                  fontWeight: '700',
                  color: '#36404D'
                }}>Motor Health</h3>
              </div>

              <span style={{
                borderRadius: '9999px',
                border: '1px solid #6299BB',
                backgroundColor: '#d0e5f1',
                padding: '0.25rem 0.75rem',
                fontSize: '0.75rem',
                color: '#36404D',
                fontWeight: '600'
              }}>
                Demo
              </span>
            </div>

            <div style={{
              marginTop: '1.5rem',
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'space-between',
              gap: '2rem'
            }}>
              <div>
                <p style={{
                  fontSize: '3rem',
                  fontWeight: '900',
                  lineHeight: '1',
                  color: '#6299BB',
                  margin: 0
                }}>82</p>
                <p style={{
                  marginTop: '0.5rem',
                  fontSize: '0.875rem',
                  color: '#36404D'
                }}>out of 100</p>
              </div>

              <div style={{ textAlign: 'right' }}>
                <p style={{
                  fontSize: '0.875rem',
                  color: '#6299BB',
                  fontWeight: '600',
                  marginBottom: '0.25rem'
                }}>Status</p>
                <p style={{
                  marginTop: '0.25rem',
                  fontSize: '0.9375rem',
                  fontWeight: '700',
                  color: '#4a9d63'
                }}>
                  Healthy
                </p>
              </div>
            </div>

            <div style={{
              marginTop: '1.25rem',
              height: '12px',
              width: '100%',
              overflow: 'hidden',
              borderRadius: '9999px',
              backgroundColor: '#d0e5f1'
            }}>
              <div
                style={{
                  height: '100%',
                  borderRadius: '9999px',
                  backgroundColor: '#4a9d63',
                  width: '82%'
                }}
              />
            </div>

            <p style={{
              marginTop: '0.75rem',
              fontSize: '0.75rem',
              color: '#6299BB'
            }}>
              
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

function Step({
  num,
  title,
  desc,
  icon,
}: {
  num: string;
  title: string;
  desc: string;
  icon?: React.ReactNode;
}) {
  return (
    <div style={{
      display: 'flex',
      gap: '1.5rem',
      borderRadius: '20px',
      border: '1px solid #d0e5f1',
      backgroundColor: '#F0F0F0',
      padding: '1.5rem',
      boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.08)'
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.75rem',
        flexShrink: 0
      }}>
        <div style={{
          display: 'flex',
          height: '48px',
          width: '48px',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '12px',
          border: '2px solid #6299BB',
          backgroundColor: '#d0e5f1',
          fontSize: '1rem',
          fontWeight: '900',
          color: '#36404D'
        }}>
          {num}
        </div>
        {icon && (
          <div style={{
            width: '80px',
            height: '80px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#ffffff',
            borderRadius: '12px',
            border: '1px solid #d0e5f1',
            padding: '0.75rem'
          }}>
            {icon}
          </div>
        )}
      </div>

      <div style={{ flex: 1 }}>
        <h3 style={{
          fontSize: '1.25rem',
          fontWeight: '700',
          color: '#36404D',
          marginBottom: '0.5rem'
        }}>{title}</h3>
        <p style={{
          marginTop: '0.5rem',
          fontSize: '0.9375rem',
          lineHeight: '1.75',
          color: '#36404D'
        }}>{desc}</p>
      </div>
    </div>
  );
}

// Icon Components for each step
function SimulinkIcon() {
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Motor/Engine representation */}
      <rect x="15" y="20" width="30" height="20" rx="4" fill="#6299BB" opacity="0.3"/>
      <circle cx="30" cy="30" r="8" fill="#6299BB"/>
      <circle cx="30" cy="30" r="4" fill="#ffffff"/>
      {/* Data waves */}
      <path d="M10 15 Q15 20 20 15 T30 15 T40 15" stroke="#6299BB" strokeWidth="2" fill="none"/>
      <path d="M10 25 Q15 30 20 25 T30 25 T40 25" stroke="#6299BB" strokeWidth="2" fill="none"/>
      <path d="M20 35 Q25 40 30 35 T40 35" stroke="#6299BB" strokeWidth="2" fill="none"/>
      {/* Output arrow */}
      <path d="M45 30 L52 30 M50 28 L52 30 L50 32" stroke="#4a9d63" strokeWidth="2" fill="none"/>
    </svg>
  );
}

function AIModelIcon() {
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Neural network nodes */}
      <circle cx="15" cy="15" r="4" fill="#6299BB"/>
      <circle cx="30" cy="10" r="4" fill="#6299BB"/>
      <circle cx="45" cy="15" r="4" fill="#6299BB"/>
      <circle cx="15" cy="30" r="4" fill="#6299BB"/>
      <circle cx="30" cy="25" r="4" fill="#6299BB"/>
      <circle cx="45" cy="30" r="4" fill="#6299BB"/>
      <circle cx="20" cy="45" r="4" fill="#6299BB"/>
      <circle cx="30" cy="40" r="4" fill="#6299BB"/>
      <circle cx="40" cy="45" r="4" fill="#6299BB"/>
      <circle cx="30" cy="52" r="4" fill="#4a9d63"/>
      {/* Connections */}
      <line x1="15" y1="15" x2="30" y2="25" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="10" x2="30" y2="25" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="45" y1="15" x2="30" y2="25" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="25" x2="20" y2="45" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="25" x2="30" y2="40" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="25" x2="40" y2="45" stroke="#6299BB" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="40" x2="30" y2="52" stroke="#4a9d63" strokeWidth="2"/>
    </svg>
  );
}

function SensorIcon() {
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Microcontroller box */}
      <rect x="18" y="25" width="24" height="18" rx="2" fill="#6299BB" opacity="0.3"/>
      <rect x="20" y="27" width="20" height="14" rx="1" fill="#6299BB"/>
      {/* Chip representation */}
      <rect x="24" y="30" width="12" height="8" fill="#ffffff"/>
      <line x1="27" y1="32" x2="33" y2="32" stroke="#6299BB" strokeWidth="0.5"/>
      <line x1="27" y1="34" x2="33" y2="34" stroke="#6299BB" strokeWidth="0.5"/>
      <line x1="27" y1="36" x2="33" y2="36" stroke="#6299BB" strokeWidth="0.5"/>
      {/* Sensors around */}
      <circle cx="12" cy="20" r="3" fill="#4a9d63"/>
      <circle cx="48" cy="20" r="3" fill="#4a9d63"/>
      <circle cx="12" cy="48" r="3" fill="#4a9d63"/>
      <circle cx="48" cy="48" r="3" fill="#4a9d63"/>
      {/* Sensor connections */}
      <line x1="15" y1="20" x2="20" y2="28" stroke="#4a9d63" strokeWidth="1.5" opacity="0.6"/>
      <line x1="45" y1="20" x2="38" y2="28" stroke="#4a9d63" strokeWidth="1.5" opacity="0.6"/>
      <line x1="15" y1="48" x2="20" y2="40" stroke="#4a9d63" strokeWidth="1.5" opacity="0.6"/>
      <line x1="45" y1="48" x2="38" y2="40" stroke="#4a9d63" strokeWidth="1.5" opacity="0.6"/>
    </svg>
  );
}

function DashboardIcon() {
  return (
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Screen */}
      <rect x="8" y="12" width="44" height="36" rx="2" fill="#6299BB" opacity="0.2"/>
      <rect x="10" y="14" width="40" height="32" rx="1" fill="#ffffff"/>
      {/* Graph bars */}
      <rect x="14" y="38" width="4" height="6" fill="#6299BB"/>
      <rect x="20" y="32" width="4" height="12" fill="#6299BB"/>
      <rect x="26" y="28" width="4" height="16" fill="#6299BB"/>
      <rect x="32" y="34" width="4" height="10" fill="#4a9d63"/>
      <rect x="38" y="30" width="4" height="14" fill="#4a9d63"/>
      <rect x="44" y="36" width="4" height="8" fill="#4a9d63"/>
      {/* Health score */}
      <text x="30" y="25" fontSize="8" fill="#6299BB" fontWeight="bold" textAnchor="middle">82</text>
      {/* Status indicator */}
      <circle cx="20" cy="24" r="2" fill="#4a9d63"/>
    </svg>
  );
}

function PipelineDiagram() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem',
      width: '100%'
    }}>
      {/* Step 1 */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: '1', minWidth: '120px' }}>
        <div style={{
          width: '70px',
          height: '70px',
          borderRadius: '12px',
          backgroundColor: '#d0e5f1',
          border: '2px solid #6299BB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '0.5rem'
        }}>
          <SimulinkIcon />
        </div>
        <p style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6299BB', textAlign: 'center' }}>Simulation</p>
      </div>

      {/* Arrow 1 */}
      <div style={{ flex: '0 0 auto', margin: '0 0.5rem' }}>
        <svg width="40" height="20" viewBox="0 0 40 20" fill="none">
          <path d="M5 10 L35 10 M30 6 L35 10 L30 14" stroke="#6299BB" strokeWidth="2.5" fill="none"/>
        </svg>
      </div>

      {/* Step 2 */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: '1', minWidth: '120px' }}>
        <div style={{
          width: '70px',
          height: '70px',
          borderRadius: '12px',
          backgroundColor: '#d0e5f1',
          border: '2px solid #6299BB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '0.5rem'
        }}>
          <AIModelIcon />
        </div>
        <p style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6299BB', textAlign: 'center' }}>AI Model</p>
      </div>

      {/* Arrow 2 */}
      <div style={{ flex: '0 0 auto', margin: '0 0.5rem' }}>
        <svg width="40" height="20" viewBox="0 0 40 20" fill="none">
          <path d="M5 10 L35 10 M30 6 L35 10 L30 14" stroke="#6299BB" strokeWidth="2.5" fill="none"/>
        </svg>
      </div>

      {/* Step 3 */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: '1', minWidth: '120px' }}>
        <div style={{
          width: '70px',
          height: '70px',
          borderRadius: '12px',
          backgroundColor: '#d0e5f1',
          border: '2px solid #6299BB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '0.5rem'
        }}>
          <SensorIcon />
        </div>
        <p style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6299BB', textAlign: 'center' }}>Sensors</p>
      </div>

      {/* Arrow 3 */}
      <div style={{ flex: '0 0 auto', margin: '0 0.5rem' }}>
        <svg width="40" height="20" viewBox="0 0 40 20" fill="none">
          <path d="M5 10 L35 10 M30 6 L35 10 L30 14" stroke="#6299BB" strokeWidth="2.5" fill="none"/>
        </svg>
      </div>

      {/* Step 4 */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: '1', minWidth: '120px' }}>
        <div style={{
          width: '70px',
          height: '70px',
          borderRadius: '12px',
          backgroundColor: '#d0e5f1',
          border: '2px solid #6299BB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '0.5rem'
        }}>
          <DashboardIcon />
        </div>
        <p style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6299BB', textAlign: 'center' }}>Dashboard</p>
      </div>
    </div>
  );
}

export default ProjectOverview;
