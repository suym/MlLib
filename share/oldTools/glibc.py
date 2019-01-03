#升级glibc (当系统版本为6.x时需要升级)
def updatelib(OSversion):
    if OSversion < float('7'):
        print"Info:[updatelib] update glibc"
        os.chdir(Python_dir + '/Installation_package/glibc-2.14.1-rpm-all')
        subprocess.call("rpm -e --nodeps --justdb glibc-2.*.i686 --allmatches &>/dev/null",shell=True)
        aa = subprocess.call("rpm -Uvh glibc-2.14.1-6.x86_64.rpm glibc-common-2.14.1-6.x86_64.rpm glibc-headers-2.14.1-6.x86_64.rpm glibc-devel-2.14.1-6.x86_64.rpm nscd-2.14.1-6.x86_64.rpm &>/dev/null",shell=True)
        if aa != 0:
            print "Warning:glib update failure!"