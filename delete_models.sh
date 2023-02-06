function delete(){
    DIR="./models/";

    echo -n "$*[Y/n]:"

    cd $DIR;

    read ANS

    case $ANS in
    "" | [Yy]* )
        # ここに「Yes」の時の処理を書く
        echo "Yes"
        rm *.pickle;
        ;;
    * )
        # ここに「No」の時の処理を書く
        echo "No"
        ;;
    esac

}

delete "全保存モデルを削除しますか？";
