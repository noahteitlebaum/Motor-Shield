import Link from "next/link";

type Props={
    name: string, 
    link: string,
    state: boolean,
    //onSelect?: () => void
}

export default function NavButton({name, link, state, /*onSelect*/}: Props){
const bgColour = state ? "h-14 bg-linear-to-r from-[#A6C0D0] to-[#6299BB] hover:to-[#6299BB]" : "bg-[#6299BB]"
const textColour = state ? "text-[#FFFFFF] hover:text-[#36404D]" : "text-[#36404D]"
    return(
    <Link href={link}>
        <div className={`flex justify-center items-center w-[293 px] h-[103 px] ${bgColour} rounded-2xl`}
            /*onClick={onSelect}*/>
            <p className={`${textColour} text-center`}>{name}</p>
        </div>
    </Link>
    )
}